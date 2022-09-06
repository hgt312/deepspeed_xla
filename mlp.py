import argparse
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

import xla_backend, xla_overrides

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
xla_overrides.pytorch_overrides()
import deepspeed
xla_overrides.deepspeed_overrides()

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
print("INIT ENGINE!!!")

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
    print("FWD+BWD!!!")
    model.step()
    xm.mark_step()
    print("MODEL STEP!!!")
    if torch.distributed.get_rank() == 0:
        print("LOSS:", loss.item())
        # print(metrics.metrics_report())
    # print_params("step={}".format(n), model)
    if n == 10: break
