# flagtree tle
"""Multi-vendor DSA (Device-Specific API) extensions.

``dsa`` is the multi-vendor namespace for device-specific primitives: the
shared API surface (``alloc``/``copy``/``pipeline``/slices/arithmetic) lives
at this level, and each vendor's private primitives live in a per-backend
sub-namespace (see ``dsa.tsingmicro``, mirroring upstream ``dsa.ascend``).

On Ascend, keep the 3.5 CommonIR DSA surface (``tile_alloc``/``tile_copy``/
``tile_subview``/``subview``/``hint``/``workspace``/``dsa.ascend`` pipe API).
Other backends keep the 3.6 multi-vendor DSA API.
"""

from triton._flagtree_backend import FLAGTREE_BACKEND

if FLAGTREE_BACKEND == "ascend":
    import sys as _sys

    from .legacy import *  # noqa: F401,F403
    from .legacy import __all__ as _legacy_all
    from .legacy import ascend, core, semantic, types
    from .legacy.ascend import core as _ascend_core
    from .legacy.ascend import custom_ops as _ascend_custom_ops
    from .legacy.ascend import pipe as _ascend_pipe

    # Keep direct submodule imports on the 3.5 implementation as well.
    for _name, _module in (("core", core), ("types", types), ("semantic", semantic), ("ascend", ascend),
                           ("ascend.core", _ascend_core), ("ascend.pipe", _ascend_pipe),
                           ("ascend.custom_ops", _ascend_custom_ops)):
        _sys.modules[__name__ + "." + _name] = _module

    __all__ = list(_legacy_all)
else:
    from .core import (
        tile_load,
        tile_store,
        pipeline,
        alloc,
        copy,
        memory_space,
        local_ptr,
        to_tensor,
        to_buffer,
        add,
        sub,
        mul,
        max,
        min,
        div,
        extract_slice,
        extract_element,
        insert_slice,
    )
    from .types import (
        scope,
        local,
        buffered_tensor,
        buffered_tensor_type,
    )
    from . import tsingmicro
    from . import thrive
    from .semantic import DSASemantic, DSASemanticError

    __all__ = [
        "tile_load",
        "tile_store",
        "pipeline",
        "alloc",
        "copy",
        "memory_space",
        "local_ptr",
        "to_tensor",
        "to_buffer",
        "add",
        "sub",
        "mul",
        "max",
        "min",
        "div",
        "extract_slice",
        "extract_element",
        "insert_slice",
        "scope",
        "local",
        "tsingmicro",
        "thrive",
        "buffered_tensor",
        "buffered_tensor_type",
        "DSASemantic",
        "DSASemanticError",
    ]
