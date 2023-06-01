r""""Contains definitions of the methods used by the _BaseDataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import paddle
from paddle.fluid.dygraph.parallel import Queue, Iterable, string_classes
from . import MP_STATUS_CHECK_INTERVAL
from paddle.fluid.dygraph.parallel import ExceptionWrapper

def _pin_memory_loop(in_queue, out_queue, device_id, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    paddle.set_num_threads(1)
    paddle.fluid.cuda_places(device_id)
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except Queue.Empty:
            continue
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = pin_memory(data)
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id))
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except Queue.Full:
                continue
        del r  # save memory

def pin_memory(data):
    if isinstance(data, paddle.Tensor):
        return data.pin_memory()
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, Iterable):
        return [pin_memory(sample) for sample in data]
    elif isinstance(data, dict):
        return {k: pin_memory(sample) for k, sample in data.items()}
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data

