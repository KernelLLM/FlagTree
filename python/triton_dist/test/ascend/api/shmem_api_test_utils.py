# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from types import SimpleNamespace

from triton_dist.language.extra.ascend import libaclshmem_device

from test_gm_addr_args_indices import _make_context, _parse, _run_pass


def capture_extern(monkeypatch):
    """Capture the Python ABI while preserving the public/backend dispatch path."""
    semantic = SimpleNamespace(
        builder=SimpleNamespace(get_string_attr=lambda value: value)
    )

    class ExternResult(str):
        def __new__(cls):
            result = super().__new__(cls, "extern_result")
            result.handle = result
            result.attrs = {}
            return result

        def set_attr(self, key, value):
            self.attrs[key] = value

    calls = []

    def cast(value, dtype, _semantic=None):
        assert _semantic is semantic
        return ("cast", value, dtype)

    def extern_call(lib_name, lib_path, args, type_map, is_pure, _semantic=None):
        assert _semantic is semantic
        assert lib_name == "libshmem_device"
        assert lib_path == ""
        assert is_pure is False
        calls.append((args, type_map))
        return ExternResult()

    monkeypatch.setattr(libaclshmem_device.tl, "cast", cast)
    monkeypatch.setattr(libaclshmem_device, "extern_call", extern_call)
    return semantic, calls


def check_extern_lowering(symbol, arg_types, result_type, core_type, collective, mixed):
    """Check custom-op ABI, effects and placement in vector and mixed kernels."""
    params = ", ".join(f"%arg{i}: {ty}" for i, ty in enumerate(arg_types))
    args = ", ".join(f"%arg{i}" for i in range(len(arg_types)))
    result = "%result = " if result_type else ""
    dot = ""
    if mixed:
        dot = """
        %lhs = arith.constant dense<0.0> : tensor<1x1xbf16>
        %rhs = arith.constant dense<0.0> : tensor<1x1xbf16>
        %acc = arith.constant dense<0.0> : tensor<1x1xf32>
        %dot = tt.dot %lhs, %rhs, %acc, inputPrecision = tf32 : tensor<1x1xbf16> * tensor<1x1xbf16> -> tensor<1x1xf32>
        """
    ctx = _make_context()
    module = _parse(
        ctx,
        f"""
        module {{
          tt.func public @kernel({params}) attributes {{noinline = false}} {{
            {dot}
            {result}"distributed.extern_call"({args}) {{libname = "libshmem_device", libpath = "", pure = false, symbol = "{symbol}", distributed.core_type = "{core_type.lower()}"}} : ({', '.join(arg_types)}) -> {result_type or '()'}
            tt.return
          }}
        }}
        """,
    )
    _run_pass(module, ctx)
    lines = [line for line in module.str().splitlines() if "hir.custom" in line]
    assert len(lines) == 1
    line = lines[0]
    assert f'symbol = "{symbol}"' in line
    assert f"hivm.tcore_type = #hivm.tcore_type<{core_type}>" in line
    assert "hivm.pipe = #hivm.pipe<PIPE_S>" in line
    assert "pure = false" in line
    assert "no_side_effect" not in line
    assert "distributed.core_type" not in line
    assert ("already_sub_block_bound" in line) == collective
    indices = [str(i) for i, ty in enumerate(arg_types) if ty.startswith("!tt.ptr")]
    expected_indices = ": " + ", ".join(indices) if indices else ""
    assert f"gm_addr_args_indices = array<i32{expected_indices}>" in line
    if result_type:
        # MLIR printers may append a source location after the result type.
        assert line.split(" loc(", 1)[0].rstrip().endswith(f"-> {result_type}")
    else:
        assert " -> " not in line


def check_scalar_extern_abi(
    monkeypatch, module, name, symbol, dtype, num_values, type_map, *, op=None, **kwargs
):
    from types import SimpleNamespace
    import triton.language as tl

    semantic, calls = capture_extern(monkeypatch)
    ptr = SimpleNamespace(dtype=tl.pointer_type(dtype))
    values = [object() for _ in range(num_values)]
    pe = tl.constexpr(1)
    args = [ptr, *values, pe]
    if op is not None:
        args = [ptr, values[0] if values else None, pe, tl.constexpr(op)]
    result = getattr(module, name)(*args, _semantic=semantic, **kwargs)
    assert result == "extern_result"
    core_type = "cube_or_vector" if name == "g" else "vector"
    assert result.attrs == {"distributed.core_type": core_type}
    assert calls == [
        (
            [ptr]
            + [("cast", value, dtype) for value in values]
            + [("cast", pe, tl.int32)],
            type_map,
        )
    ]
    signature = (tl.pointer_type(dtype),) + (dtype,) * num_values + (tl.int32,)
    assert calls[0][1][signature][0] == symbol


def check_unsupported_dtype(monkeypatch, name, dtype, num_values, *, op=None, **kwargs):
    from types import SimpleNamespace
    import pytest
    import triton.language as tl

    def unexpected_call(*args):
        pytest.fail("An unsupported dtype must not emit an extern call")

    semantic = SimpleNamespace(
        to_tensor=lambda value: value,
        builder=SimpleNamespace(create_extern_call=unexpected_call),
    )
    monkeypatch.setattr(
        tl, "cast", lambda value, dtype, _semantic=None: tl.tensor(None, dtype)
    )
    ptr = tl.tensor(None, tl.pointer_type(dtype))
    values = [tl.tensor(None, tl.int32) for _ in range(num_values)]
    with pytest.raises(ValueError, match="input arg type does not match"):
        args = [ptr, *values, 1]
        if op is not None:
            args = [ptr, values[0] if values else None, 1, tl.constexpr(op)]
        getattr(libaclshmem_device, name)(*args, _semantic=semantic, **kwargs)
