import os
from collections import OrderedDict
import json
import paddle
import numpy as np
import json

def swap_to_cpu(tensor):
    tensor_cpu = paddle.to_tensor(tensor.numpy())
    return tensor_cpu

def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying paddle runtime"""
    allocated = paddle.fluid.core._cuda_memory_manager().allocated()
    reserved = paddle.fluid.core._cuda_memory_manager().reserved()
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated

def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    ret = 0
    for x in tensors:
        if x.dtype in [paddle.float32, paddle.int32]:
            ret += np.prod(x.shape) * 4
        elif x.dtype in [paddle.bfloat16, paddle.float16, paddle.int16]:
            ret += np.prod(x.shape) * 2
        elif x.dtype in [paddle.int8]:
            ret += np.prod(x.shape) * 1
    return ret

def empty_cache(ratio):
    if ratio is None:
        return
    allocated = paddle.fluid.core._cuda_memory_manager().allocated()
    reserved = paddle.fluid.core._cuda_memory_manager().reserved()
    if reserved > 0 and allocated / reserved < ratio:
        paddle.fluid.core._cuda_memory_manager().empty_cache()

def disable_cache_allocator():
    os.environ['PADDLE_NO_CUDA_MEMORY_CACHING'] = '1'

def enable_cache_allocator():
    del os.environ['PADDLE_NO_CUDA_MEMORY_CACHING']

class GlobalExpRecorder:
    def __init__(self):
        self.val_dict = OrderedDict()
    def record(self, key, value, float_round=6):
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)
        if isinstance(value, (float, np.float32, np.float64)):
            value = round(value, float_round)
        self.val_dict[key] = value
    def dump(self, filename):
        with open(filename, "a") as fout:
            fout.write(json.dumps(self.val_dict) + '\n')
        print("Save exp results to %s" % filename)

    def clear(self):
        pass

exp_recorder = GlobalExpRecorder()

