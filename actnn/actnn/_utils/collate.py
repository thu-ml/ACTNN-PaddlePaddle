r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).
These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import paddle
import re
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.container import MapContainer, TupleContainer, ListContainer
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, paddle.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return to_variable(data)
    elif isinstance(data, MapContainer):
        return MapContainer({key: default_convert(data[key]) for key in data})
    elif isinstance(data, TupleContainer) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, ListContainer) and not isinstance(data, string_classes):
        return ListContainer([default_convert(d) for d in data])
    else:
        return data
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, paddle.Tensor):
        out = None
        if paddle.fluid.dygraph.parallel.Env().nranks > 1:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = paddle.fluid.core.PaddleTensorStorage(numel)
            out = paddle.fluid.core.PaddleTensor(elem.dtype)
            out.set_dims([numel])
            out.set_layout(paddle.fluid.core.VarDesc.VarLayout.LOD_TENSOR)
            out.set_tensor_from_storage(storage)
            out = to_variable(out)
        return paddle.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([to_variable(b) for b in batch])
        elif elem.shape == ():  # scalars
            return to_variable(batch)
    elif isinstance(elem, float):
        return paddle.to_tensor(batch, dtype='float64')
    elif isinstance(elem, int_classes):
        return paddle.to_tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, MapContainer):
        return MapContainer({key: default_collate([d[key] for d in batch]) for key in elem})
    elif isinstance(elem, TupleContainer) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, ListContainer):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return ListContainer([default_collate(samples) for samples in transposed])
    raise TypeError(default_collate_err_msg_format.format(elem_type))

