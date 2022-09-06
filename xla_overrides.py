import contextlib


def pytorch_overrides():
    import torch
    print("pytorch overide for deepspeed")

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
    torch.cuda.synchronize = torch.distributed.barrier

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
            torch.distributed.barrier()
        def wait_stream(self, stream):
            torch.distributed.barrier()
    torch.cuda.Stream = DummyStream
    @contextlib.contextmanager
    def dummy_ctx(_): yield
    torch.cuda.stream = dummy_ctx
    torch.cuda.current_stream = lambda: DummyStream()
    torch.cuda.default_stream = lambda: DummyStream()

    old_record_stream = torch.Tensor.record_stream
    def my_record_stream(self, stream):
        torch.distributed.barrier()
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

    torch.Tensor.data_ptr = lambda self: 0


def deepspeed_overrides():
    # merge some small graphs
    import torch_xla.core.xla_model as xm
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    origin_defragment_func = DeepSpeedZeroOptimizer_Stage3.defragment
    def new_defragment(tensors):
        xm.unlazy(tensors)
        return origin_defragment_func(tensors)
    DeepSpeedZeroOptimizer_Stage3.defragment = new_defragment
