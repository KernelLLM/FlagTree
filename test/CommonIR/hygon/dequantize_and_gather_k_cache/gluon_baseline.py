import sys as _sys
if _sys.flags.no_site:
    import sysconfig as _sysconfig
    from pathlib import Path as _Path
    _python_version = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
    for _package_path in dict.fromkeys((
        _sysconfig.get_path("purelib"),
        _sysconfig.get_path("platlib"),
        f"/usr/local/lib/{_python_version}/dist-packages",
        f"/usr/lib/{_python_version}/dist-packages",
        "/usr/lib/python3/dist-packages",
    )):
        if _package_path and _Path(_package_path).is_dir() and _package_path not in _sys.path:
            _sys.path.append(_package_path)

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

NUM_WARPS = 4
NUM_STAGES = 2


@g.jit
def _dequantize_and_gather_k_cache_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: gl.constexpr,
    nope_dim: gl.constexpr,
    rope_dim: gl.constexpr,
    scale_slots: gl.constexpr,
    quant_block: gl.constexpr,
    cache_block_size: gl.constexpr,
    token_data_size: gl.constexpr,
    cache_block_stride: gl.constexpr,
    output_dim: gl.constexpr,
    num_workers: gl.constexpr,
    n_quant_blocks: gl.constexpr,
    HAVE_GATHER_LENS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    req_idx = gl.program_id(0)
    worker_idx = gl.program_id(1)
    seq_len = gl.load(seq_lens_ptr + req_idx)
    if HAVE_GATHER_LENS:
        gather_len = gl.load(gather_lens_ptr + req_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for local_i in range(worker_idx, gather_len, num_workers):
        pos = start_pos + local_i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos - block_in_seq * cache_block_size
        physical_block = gl.load(
            block_table_ptr + req_idx * max_blocks_per_seq + block_in_seq
        )
        cache_block = k_cache_ptr + physical_block.to(gl.int64) * cache_block_stride
        token_data = cache_block + pos_in_block * token_data_size
        scale_base = (
            cache_block
            + cache_block_size * token_data_size
            + pos_in_block * scale_slots
        )
        out_row = out_ptr + req_idx * out_stride0 + (offset + local_i) * out_stride1

        if nope_dim % quant_block == 0:
            for qblock in gl.static_range(0, n_quant_blocks):
                qoffs = qblock * quant_block + gl.arange(0, quant_block, layout=layout)
                x_u8 = gl.load(token_data + qoffs)
                x_fp8 = x_u8.to(gl.float8e4nv, bitcast=True).to(gl.float32)
                encoded = gl.load(scale_base + qblock)
                scale = gl.exp2(encoded.to(gl.float32) - 127.0)
                x = x_fp8 * scale
                gl.store(out_row + qoffs, x.to(gl.bfloat16))
        else:
            for qblock in gl.static_range(0, n_quant_blocks - 1):
                qoffs = qblock * quant_block + gl.arange(0, quant_block, layout=layout)
                x_u8 = gl.load(token_data + qoffs)
                x_fp8 = x_u8.to(gl.float8e4nv, bitcast=True).to(gl.float32)
                encoded = gl.load(scale_base + qblock)
                scale = gl.exp2(encoded.to(gl.float32) - 127.0)
                x = x_fp8 * scale
                gl.store(out_row + qoffs, x.to(gl.bfloat16))

            qblock = n_quant_blocks - 1
            qoffs = qblock * quant_block + gl.arange(0, quant_block, layout=layout)
            qmask = qoffs < nope_dim
            x_u8 = gl.load(token_data + qoffs, mask=qmask, other=0)
            x_fp8 = x_u8.to(gl.float8e4nv, bitcast=True).to(gl.float32)
            encoded = gl.load(scale_base + qblock)
            scale = gl.exp2(encoded.to(gl.float32) - 127.0)
            x = x_fp8 * scale
            gl.store(out_row + qoffs, x.to(gl.bfloat16), mask=qmask)

        bf16_ptr = (token_data + nope_dim).to(gl.pointer_type(gl.bfloat16))
        for rblock in gl.static_range(0, rope_dim, 64):
            roffs = rblock + gl.arange(0, 64, layout=layout)
            rmask = roffs < rope_dim
            vals = gl.load(bf16_ptr + roffs, mask=rmask, other=0.0)
            gl.store(out_row + nope_dim + roffs, vals, mask=rmask)


def dequantize_and_gather_k_cache(
    out,
    k_cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size: int,
    offset: int = 0,
    rope_dim: int = 64,
    nope_dim: int | None = None,
    scale_slots: int | None = None,
) -> None:
    """Launch the historical contiguous-cache GPU flow, without hidden work.

    The accepted cases use contiguous BF16 output and contiguous uint8 cache.
    Validation of device-resident lengths/indices is the caller's responsibility.
    No allocation, cache packing, or input conversion occurs in this wrapper.
    """
    if out.ndim != 3 or str(out.dtype) != "torch.bfloat16" or not out.is_contiguous():
        raise ValueError("out must be a contiguous three-dimensional BF16 tensor")
    if k_cache.ndim not in (2, 3) or not k_cache.is_contiguous() or str(k_cache.dtype) != "torch.uint8":
        raise ValueError("k_cache must be contiguous uint8, two- or three-dimensional")
    if block_table.ndim != 2 or not block_table.is_contiguous():
        raise ValueError("block_table must be contiguous and two-dimensional")
    if seq_lens.ndim != 1 or seq_lens.shape[0] != block_table.shape[0]:
        raise ValueError("request dimensions must agree")
    if not 0 < block_size or not 0 <= offset:
        raise ValueError("block_size must be positive and offset nonnegative")
    output_dim = out.shape[-1]
    if nope_dim is None:
        nope_dim = output_dim - rope_dim
    if nope_dim <= 0 or rope_dim < 0 or nope_dim % 2 or nope_dim + rope_dim > output_dim:
        raise ValueError("invalid FP8/RoPE dimensions or BF16 byte alignment")
    if scale_slots is None:
        scale_slots = triton.cdiv(nope_dim, 64) + (1 if nope_dim % 64 == 0 else 0)
    if scale_slots < triton.cdiv(nope_dim, 64):
        raise ValueError("scale_slots does not cover all quantization blocks")
    cache = k_cache.view(k_cache.shape[0], -1)
    workers = 128
    _dequantize_and_gather_k_cache_kernel[(seq_lens.shape[0], workers)](
        out, out.stride(0), out.stride(1), cache, seq_lens, block_table,
        offset, gather_lens, block_table.shape[-1], nope_dim, rope_dim,
        scale_slots, 64, block_size, nope_dim + rope_dim * 2, cache.stride(0),
        output_dim, workers, triton.cdiv(nope_dim, 64), gather_lens is not None,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )


def _self_test():
    """One small_aligned case, independent finite E4M3FN mathematical decoding."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("a compatible HIP/DCU environment is required")
    torch.manual_seed(7129)
    block_size, seq_len, count, nope_dim, rope_dim = 4, 6, 3, 64, 16
    scale_slots = 2
    token_bytes = nope_dim + 2 * rope_dim
    cache_stride = block_size * (token_bytes + scale_slots)
    cpu_cache = torch.zeros((2, cache_stride), dtype=torch.uint8)
    tokens = cpu_cache[:, :block_size * token_bytes].view(2, block_size, token_bytes)
    tokens[..., :nope_dim] = torch.randint(0, 126, (2, block_size, nope_dim), dtype=torch.uint8)
    rope = torch.randn((2, block_size, rope_dim)).to(torch.bfloat16)
    tokens[..., nope_dim:] = rope.contiguous().view(torch.uint8).reshape(2, block_size, 2 * rope_dim)
    cpu_cache[:, block_size * token_bytes:] = 127
    expected = torch.empty((1, count, nope_dim + rope_dim), dtype=torch.bfloat16)
    for i, pos in enumerate(range(seq_len - count, seq_len)):
        block, offset = divmod(pos, block_size)
        raw = tokens[block, offset]
        code = raw[:nope_dim].int()
        exponent, mantissa = (code >> 3) & 15, (code & 7).float()
        magnitude = torch.where(exponent == 0, mantissa * (2.0 ** -9),
                                torch.pow(2.0, exponent.float() - 7) * (1.0 + mantissa / 8.0))
        expected[0, i, :nope_dim] = torch.where((code & 128) != 0, -magnitude, magnitude).to(torch.bfloat16)
        expected[0, i, nope_dim:] = raw[nope_dim:].contiguous().view(torch.bfloat16)
    out = torch.empty_like(expected, device="cuda")
    cache = cpu_cache.to("cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([count], dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    dequantize_and_gather_k_cache(out, cache, seq_lens, gather_lens, block_table,
                                block_size, rope_dim=rope_dim, nope_dim=nope_dim,
                                scale_slots=scale_slots)
    torch.cuda.synchronize()
    if not torch.equal(out.cpu().view(torch.uint8), expected.view(torch.uint8)):
        raise AssertionError("dequantized output differs bitwise from independent reference")
    variant = "baseline" if NUM_WARPS == 4 else "optimized"
    print(f"PASS dequantize_and_gather_k_cache {variant}: small_aligned (bitwise)", flush=True)


if __name__ == "__main__":
    _self_test()
