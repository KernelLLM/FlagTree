import os
from typing import List, Union

from triton.language import core as tl
from triton._C.libtriton import ir

from triton.experimental.tle.language.dsa import semantic as tile_semantic
from triton.experimental.tle.language.dsa.types import buffer, buffer_type
from .types import address_space


def _cuda_execution_mode() -> bool:
    return os.environ.get("TLE_GPU_EXECUTION_MODE") == "1"


def _make_symbolic_buffer(etype: tl.dtype, shape, space: address_space) -> buffer:
    shape = tl._unwrap_shape(shape)
    return buffer(None, buffer_type(element_ty=etype, shape=shape, space=space))


def alloc(etype: tl.dtype, shape: List[tl.constexpr], space: address_space, builder: ir.builder) -> buffer:
    if _cuda_execution_mode():
        return _make_symbolic_buffer(tl._constexpr_to_value(etype), shape, space)
    return tile_semantic.tile_alloc(etype, shape, space, builder)


def copy(src, dst: buffer, shape: List[Union[tl.constexpr, int]], inter_no_alias: bool, builder: ir.builder):
    if not _cuda_execution_mode():
        tile_semantic.tile_copy(src, dst, shape, inter_no_alias, builder)
    # Keep a frontend-only association so the NVIDIA path can reuse the normal
    # tl.load -> TTGIR pipeline while the TileIR dump still contains tile.copy.
    if isinstance(dst, buffer):
        dst._tle_gpu_last_copy_src = src
        dst._tle_gpu_last_copy_shape = shape


def to_tensor(memref: buffer, writable: bool, builder: ir.builder, target_shape=None) -> tl.tensor:
    tile_value = None
    if not _cuda_execution_mode():
        tile_value = tile_semantic.tile_to_tensor(memref, writable, builder, target_shape=target_shape)
        return tile_value
    src = getattr(memref, "_tle_gpu_last_copy_src", None)
    if src is not None:
        return tl.load(src, _builder=builder)
    tensor_value = getattr(memref, "_tle_gpu_tensor_value", None)
    if tensor_value is not None:
        return tensor_value
    if tile_value is not None:
        return tile_value
    raise ValueError("tle.gpu.to_tensor needs a preceding tle.gpu.copy or tle.gpu.store_tensor in CUDA mode")


def store_tensor(tensor: tl.tensor, dst: buffer, builder: ir.builder):
    if not _cuda_execution_mode():
        tile_semantic.tile_store_tensor(tensor, dst, builder)
    if isinstance(dst, buffer):
        dst._tle_gpu_tensor_value = tensor


def to_buffer(tensor: tl.tensor, space: address_space, bind_buffer: buffer, builder: ir.builder) -> buffer:
    if _cuda_execution_mode():
        dst = bind_buffer
        if dst is None:
            dst = _make_symbolic_buffer(tensor.dtype, tensor.shape, space)
        dst._tle_gpu_tensor_value = tensor
        return dst
    dst = tile_semantic.tile_to_buffer(tensor, space, bind_buffer, builder)
    dst._tle_gpu_tensor_value = tensor
    return dst


def subview(src: buffer, offsets: List[tl.tensor], sizes: List[tl.constexpr], strides: List[tl.constexpr],
            builder: ir.builder) -> buffer:
    if _cuda_execution_mode():
        result = _make_symbolic_buffer(src.dtype, sizes, src.space)
    else:
        result = tile_semantic.tile_subview(src, offsets, sizes, strides, builder)
    if hasattr(src, "_tle_gpu_last_copy_src"):
        result._tle_gpu_last_copy_src = src._tle_gpu_last_copy_src
        result._tle_gpu_last_copy_shape = getattr(src, "_tle_gpu_last_copy_shape", sizes)
    if hasattr(src, "_tle_gpu_tensor_value"):
        result._tle_gpu_tensor_value = src._tle_gpu_tensor_value
    return result
