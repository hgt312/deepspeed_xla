import os
import random
import contextlib

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import transformers

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
# os.environ["XLA_SAVE_TENSORS_FILE"] = "bert_xla_log.txt"

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

####################  script  ####################
def check_batch_size(inputs, expected_batch_size):
    """Check if it is the last batch."""
    return inputs.shape[0] == expected_batch_size


def load_wikitext2(batch_size):
    """Download (if needed) and load WikiText-2 dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    max_len = 128

    print("Loading WikiText-2...")
    # TODO: shuffle
    datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported
        # it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_len) * max_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_len] for i in range(0, total_length, max_len)]
            for k, t in concatenated_examples.items()
        }
        result["input_ids"] = torch.LongTensor(result["input_ids"])
        result["labels"] = result["input_ids"].clone().numpy()
        if result["input_ids"].shape[0] == 0:
            return result

        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(result["input_ids"].shape)
        # where the random array is less than 0.15, we set true
        mask_arr = (rand < 0.15) * (result["input_ids"] != 101) * (result["input_ids"] != 102)
        # create selection from mask_arr
        selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
        # apply selection index to inputs.input_ids, adding MASK tokens
        result["input_ids"][:, selection] = 103

        result["input_ids"] = result["input_ids"].numpy()

        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
    )

    lm_datasets["train"].set_format(type="torch", columns=["input_ids", "labels"])
    lm_datasets["validation"].set_format(
        type="torch", columns=["input_ids", "labels"]
    )
    train_loader = torch.utils.data.DataLoader(lm_datasets["train"], batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(lm_datasets["validation"], batch_size=batch_size)

    return train_loader, val_loader


seq_length = 128
batch_size = 64

torch.manual_seed(2222)
np.random.seed(2222)
random.seed(2222)

class ConvertNLPContext:
    """The context to deal with TOKENIZERS_PARALLELISM."""

    def __init__(self):
        self.tokenizers_parallelism = None

    def __enter__(self):
        if "TOKENIZERS_PARALLELISM" in os.environ:
            self.tokenizers_parallelism = os.environ["TOKENIZERS_PARALLELISM"]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def __exit__(self, ptype, value, trace):
        if self.tokenizers_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = self.tokenizers_parallelism
        else:
            del os.environ["TOKENIZERS_PARALLELISM"]

with ConvertNLPContext():
    config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
    config.use_cache = False  # Disable model cache to avoid unnecessary model outputs.
    config.vocab_size = 32032
    config.num_hidden_layers = 1
    model = transformers.BertForMaskedLM(config)

    input_shape = [batch_size, seq_length]

    np_x = np.random.randint(0, 10000, input_shape)
    t_x = torch.tensor(np_x)
    t_y = model(t_x)[0]
    torch.cuda.empty_cache()

model.train()

torch.manual_seed(2222 + rank)
np.random.seed(2222 + rank)
random.seed(2222 + rank)

trainloader, testloader = load_wikitext2(batch_size)

deepspeed_overrides()
import deepspeed

ds_config = {
    "train_micro_batch_size_per_gpu": batch_size,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
            "torch_adam": True,
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
dist.init_process_group("xla", rank=rank, world_size=world_size)
model, _, _,_ = deepspeed.initialize(model=model,
                                     model_parameters=model.parameters(),
                                     config=ds_config)
xm.mark_step()

n_epoch = 10
print_period = 1

for epoch in range(n_epoch):
    n_iter = 1
    for batch in trainloader:
        inputs, labels = batch["input_ids"], batch["labels"]
        # Throw away the last batch if its size is smaller than the expected batch size.
        if not check_batch_size(inputs, batch_size):
            break

        inputs = inputs.to(device=device)
        labels = labels.to(device=device)

        t_y = model(inputs)
        t_y = t_y["logits"].view((batch_size * seq_length, -1))
        labels = torch.flatten(labels)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        loss = torch.nn.functional.nll_loss(t_ypred, labels)
        model.backward(loss)
        xm.mark_step()
        model.step()
        xm.mark_step()

        if torch.distributed.get_rank() == 0 and n_iter % print_period == 0:
            print("[{}, {}] loss: {:.8f}".format(epoch + 1, n_iter, loss.item()))
        n_iter += 1
