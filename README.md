# deepspeed_xla

By overriding something in PyTorch, we can make DeepSpeed treat XLA tensor as CUDA tensor, so that we can use torch_xla to run DeepSpeed models.

## Environment

- PyTorch (latest)
- DeepSpeed with minor changes (0.7.0)
- torch-xla (latest)

## Usage

1. Download this repo
2. Apply the code changes in `patches` to DeepSpeed:
    - deepspeed.patch
3. run the scripts

## Contents

- xla_backennd.py: it's just origin [xla_backend.py](https://github.com/pytorch/xla/blob/master/torch_xla/distributed/xla_backend.py) with some additional code pieces
- mlp_origin.py: origin DeepSpeed script for a single layer MLP
- mlp.py: modified `mlp_origin.py` which can run one XLA
- bert_origin.py: origin DeepSpeed script for a small BERT model
- bert.py: modified `bert_origin.py` which can run one XLA

## Run

```bash
# NCCL_P2P_DISABLE=1 XLA_IO_THREAD_POOL_SIZE=1 XLA_RNG_BIT_GENERATOR=three_fry
deepspeed mlp.py
```

You can run this command to compare the results:

```bash
deepspeed mlp_origin.py
```

For Bert model, see `bert.py` and `bert_origin.py`

## Debug

```bash
export XLA_SAVE_TENSORS_FILE=mlp_xla_log.txt
export XLA_IR_DEBUG=1
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=tensor=5,computation_client=5,xrt_computation_client=5,aten_xla_type=1 NCCL_P2P_DISABLE=1 XLA_IO_THREAD_POOL_SIZE=1 XLA_RNG_BIT_GENERATOR=three_fry deepspeed [mlp.py/bert.py]
```
