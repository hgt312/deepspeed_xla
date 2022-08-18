import argparse
import contextlib
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as metrics

import xla_backend

####################  setup xla dist config  ####################
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

os.environ[xenv.WORLD_SIZE] = str(world_size)
os.environ[xenv.ORDINAL] = str(rank)
os.environ[xenv.LOCAL_ORDINAL] = str(rank)
os.environ[xenv.LOCAL_WORKER] = "localservice:" + str(rank)
os.environ[xenv.MP_DEVICE] = "GPU:" + str(rank)
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

workers = []
for wid in range(0, world_size):
    workers.append("localservice:{};grpc://localhost:{}".format(wid, 49500 + rank))
os.environ[xenv.WORKERS] = "|".join(workers)

devices = []
for i in range(0, world_size):
    tfdevice = "/job:localservice/replica:0/task:{}/device:XLA_GPU:0".format(i)
    devices.append("GPU:{};{}".format(i, tfdevice))
os.environ[xenv.DEVICE_MAP] = "|".join(devices)

os.environ[xenv.SERVICE_ADDRESS] = "localhost:{}".format(49499)

device = xm.xla_device()
xm.set_replication(device, [device])

# os.environ["XLA_IR_DEBUG"] = "1"
# os.environ["XLA_SAVE_TENSORS_FILE"] = "mlp_xla_log.txt"

####################  deepspeed overide  ####################
def deepspeed_overrides():
    print("deepspeed_overrides")

    # override `torch.Tensor.type`
    old_tensor_type_func = torch.Tensor.type
    def my_tensor_type_func(self, dtype=None, non_blocking=False, **kwargs):
        if self.device.type == "xla" and dtype is None:
            type_map = {
                "torch.float32": "torch.cuda.FloatTensor",
                "torch.float16": "torch.cuda.HalfTensor"
            }
            return type_map[str(self.dtype)]
        return old_tensor_type_func(self, dtype, non_blocking, **kwargs)
    torch.Tensor.type = my_tensor_type_func

    # DeepSpeed hard coded torch.cuda.synchronize in their timer:
    # https://github.com/microsoft/DeepSpeed/blob/32e85eda58c560e5d5a596f71fce3682ac2ef80a/deepspeed/utils/timer.py#L141
    torch.cuda.synchronize = dist.barrier

    # Set torch.cuda.set_device to no-op
    torch.cuda.set_device = lambda x: None
    torch.cuda.current_device = lambda: "xla"
    torch.cuda.reset_peak_memory_stats = lambda: None

    class DummyEvent(object):
        def __init__(self, enable_timing=False, blocking=False, interprocess=False):
            pass
        def record(self, stream=None):
            pass
        def query(self):
            return True

    # The cuda.Event objects are never actually used in Deepspeed.
    torch.cuda.Event = DummyEvent

    # We treat cuda stream as no-op.
    class DummyStream(object):
        def __init__(self):
            pass
        def synchronize(self):
            dist.barrier()
        def wait_stream(self, stream):
            dist.barrier()
    torch.cuda.Stream = DummyStream
    @contextlib.contextmanager
    def dummy_ctx(_): yield
    torch.cuda.stream = dummy_ctx
    torch.cuda.current_stream = lambda: DummyStream()
    torch.cuda.default_stream = lambda: DummyStream()

    old_record_stream = torch.Tensor.record_stream
    def my_record_stream(self, stream):
        dist.barrier()
    torch.Tensor.record_stream = my_record_stream

    # cuda memory
    torch.cuda.memory_allocated = lambda: 0
    torch.cuda.max_memory_allocated = lambda: 0
    delattr(torch.cuda, "memory_stats")

    # typed tensors
    torch.cuda.ByteTensor = lambda t: torch.ByteTensor(t).to(device="xla")
    torch.cuda.FloatTensor = lambda t: torch.FloatTensor(t).to(device="xla")

    # Currently xla device tensor cannot be moved to CPU. This need to be fixed in
    # https://github.com/pytorch/pytorch/blob/8ad584823fd5f6ac356f87939ccd327062ca1c2f/c10/core/TensorImpl.h#L1455
    torch.Tensor.cpu = lambda self: self

    # To us, cuda device means xla device.
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self.to(device="xla")
    orig_torch_device = torch.device
    def my_device(*args):
        if len(args) == 1 and isinstance(args[0], str):
            if args[0].startswith("cuda"):
                return orig_torch_device("xla")
        elif len(args) == 2 and isinstance(args[0], str) and args[0] == "cuda" and isinstance(args[1], int):
            idx = args[1]
            return orig_torch_device("xla")
        return orig_torch_device(*args)
    torch.device = my_device

    # merge some small graphs
    # from deepspeed.runtime.zero.stage3 import FP16_DeepSpeedZeroOptimizer_Stage3
    # origin_defragment_func = FP16_DeepSpeedZeroOptimizer_Stage3.defragment
    # def new_defragment(tensors):
    #     xm.unlazy(tensors)
    #     return origin_defragment_func(tensors)
    # FP16_DeepSpeedZeroOptimizer_Stage3.defragment = new_defragment

deepspeed_overrides()
import deepspeed

####################  model  ####################
class SimpleModel(nn.Module):
    def __init__(self, hidden_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden = x
        hidden = self.linear(hidden)
        # hidden = self.linear2(hidden)
        return self.cross_entropy_loss(hidden, y)

####################  script  ####################
dist.init_process_group("xla", rank=rank, world_size=world_size)

def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, "temp_config.json")
    with open(config_path, "w") as fd:
        json.dump(config_dict, fd)
    return config_path


def get_data_loader(model, total_samples, hidden_dim):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device="cpu", dtype=torch.half)
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device="cpu").random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader


def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()  # args=""

    config_path = create_config_from_dict(tmpdir, config_dict)
    args.deepspeed_config = config_path
    return args


def print0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg, flush=True)

config_dict = {
    "train_batch_size": 8,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "torch_adam": True,
            "lr": 0.00015,
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000
    },
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": False,
        "stage3_param_persistence_threshold": 0
    }
}
args = get_args("/tmp/", config_dict)
hidden_dim = 4

print("seed:", 2222)
torch.random.manual_seed(2222)
model = SimpleModel(hidden_dim)
torch.random.manual_seed(2222 + rank)

model, _, _, _ = deepspeed.initialize(args=args,
                                      model=model,
                                      model_parameters=model.parameters())
xm.mark_step()
# print("INIT ENGINE!!!")

def print_params(tag, model):
    if torch.distributed.get_rank() == 0:
        for n, p in model.named_parameters():
            print0("{} {}:{}".format(tag, n, p))


data_loader = get_data_loader(model=model,
                              total_samples=1000,
                              hidden_dim=hidden_dim)
# print_params("pre-train", model)
for n, batch in enumerate(data_loader):
    x = batch[0].to(device=device)
    y = batch[1].to(device=device)
    loss = model(x, y)
    model.backward(loss)
    xm.mark_step()
    # print("FWD+BWD!!!")
    model.step()
    xm.mark_step()
    # print("MODEL STEP!!!")
    if torch.distributed.get_rank() == 0:
        print("LOSS:", loss.item())
        # print(metrics.metrics_report())
    # print_params("step={}".format(n), model)
    if n == 10: break
