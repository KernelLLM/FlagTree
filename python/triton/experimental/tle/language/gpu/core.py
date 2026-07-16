from functools import wraps
from typing import List, TypeVar

import triton.language.core as tl
from triton.language import semantic as tl_semantic
from triton.language.core import _constexpr_to_value

from triton.experimental.tle.language.dsa.types import buffer
from . import semantic as gpu_semantic
from .types import address_space, global_space, shared, local, register

T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"
TLE_BUILTIN = "__tle_builtin__"


def builtin(fn: T) -> T:
    assert callable

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, TLE_BUILTIN, True)
    return wrapper


@builtin
def alloc(shape: List[tl.constexpr], dtype: tl.dtype, mem_addr_space: address_space = shared,
          _builder=None) -> buffer:
    assert mem_addr_space is not None
    return gpu_semantic.alloc(dtype, shape, mem_addr_space, _builder)


@builtin
def copy(src, dst: buffer, shape, inter_no_alias=False, _builder=None):
    assert len(shape) != 0, "Can't deduce copy extents from args"
    shape = _constexpr_to_value(shape)
    inter_no_alias = _constexpr_to_value(inter_no_alias)
    gpu_semantic.copy(src, dst, shape, inter_no_alias, _builder)


@builtin
def to_tensor(memref: buffer, writable: bool = True, target_shape=None, _builder=None) -> tl.tensor:
    writable = _constexpr_to_value(writable)
    return gpu_semantic.to_tensor(memref, writable, _builder, target_shape=target_shape)


@builtin
def store_tensor(tensor: tl.tensor, dst: buffer, _builder=None):
    gpu_semantic.store_tensor(tensor, dst, _builder)


@builtin
def to_buffer(tensor: tl.tensor, space: address_space = shared, bind_buffer: buffer = None,
              _builder=None) -> buffer:
    return gpu_semantic.to_buffer(tensor, space, bind_buffer, _builder)


@builtin
def subview(src: buffer, offsets: List, sizes: List[tl.constexpr], strides: List[tl.constexpr],
            _builder=None) -> buffer:
    new_sizes = []
    for i, size in enumerate(sizes):
        if isinstance(size, int):
            new_sizes.append(tl.constexpr(size))
        elif isinstance(size, tl.constexpr):
            new_sizes.append(size)
        else:
            raise TypeError(f"sizes[{i}] must be constexpr, got {type(size).__name__}")

    new_strides = []
    for i, stride in enumerate(strides):
        if isinstance(stride, int):
            new_strides.append(tl.constexpr(stride))
        elif isinstance(stride, tl.constexpr):
            new_strides.append(stride)
        else:
            raise TypeError(f"strides[{i}] must be constexpr, got {type(stride).__name__}")

    new_offsets = []
    for offset in offsets:
        if isinstance(offset, tl.constexpr):
            if offset < 0:
                raise ValueError(f"Offset value must be non-negative, got {offset}")
            new_offsets.append(tl_semantic.to_tensor(offset, _builder))
        elif isinstance(offset, int):
            if offset < 0:
                raise ValueError(f"Offset value must be non-negative, got {offset}")
            new_offsets.append(tl_semantic.to_tensor(tl.constexpr(offset), _builder))
        else:
            new_offsets.append(offset)

    return gpu_semantic.subview(src, new_offsets, new_sizes, new_strides, _builder)
