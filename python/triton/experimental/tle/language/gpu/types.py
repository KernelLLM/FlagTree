from triton._C.libtriton import ir
import triton.language.core as tl

from triton.experimental.tle.language.dsa.types import buffer, buffer_type


class address_space:
    """GPU memory-space marker carried by !tile.buf."""

    def __init__(self, name: str):
        self.name = name

    def to_ir(self, builder: ir.builder) -> ir.attribute:
        if hasattr(builder, "dsa_get_string_attr"):
            return builder.dsa_get_string_attr(self.name)
        if hasattr(builder, "get_str_attr"):
            return builder.get_str_attr(self.name)
        return builder.dsa_get_null_attr()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, address_space) and self.name == other.name


global_space = address_space("global")
shared = address_space("shared")
local = address_space("local")
register = address_space("register")


def make_buffer(handle, element_ty: tl.dtype, shape, space: address_space, strides=None) -> buffer:
    return buffer(handle, buffer_type(element_ty=element_ty, shape=shape, space=space, strides=strides))
