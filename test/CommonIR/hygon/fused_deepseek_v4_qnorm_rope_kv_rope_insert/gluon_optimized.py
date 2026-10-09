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

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

DATA_LAYOUT = gl.BlockedLayout([8], [64], [1], [0])
ROPE_LAYOUT = gl.BlockedLayout([1], [64], [1], [0])
Q_STORE_LAYOUT = DATA_LAYOUT
LAYOUT_SPEC = {
    "data": {"sizePerThread": [8], "threadsPerWarp": [64], "warpsPerCTA": [1], "order": [0]},
    "rope": {"sizePerThread": [1], "threadsPerWarp": [64], "warpsPerCTA": [1], "order": [0]},
}
LAUNCH = {"num_warps": 1, "num_stages": 4, "waves_per_eu": 1}


@g.jit
def qnorm_rope_insert(
    q_ptr, kv_ptr, k_cache_ptr, slot_mapping_ptr, position_ids_ptr,
    cos_sin_cache_ptr, stride_q_tok, stride_q_head, stride_kv_tok,
    stride_cache_block, stride_cache_token, stride_cos_sin_pos, eps,
    num_q_items, total_items, NUM_HEADS: gl.constexpr,
    CACHE_BLOCK_SIZE: gl.constexpr,
):
    pid = gl.program_id(0)
    item_id = pid
    while item_id < total_items:
        if item_id < num_q_items:
            tok_idx = item_id // NUM_HEADS
            head_idx = item_id % NUM_HEADS
            base = tok_idx * stride_q_tok + head_idx * stride_q_head
            offs = gl.arange(0, 512, layout=DATA_LAYOUT)
            x = gl.load(q_ptr + base + offs).to(gl.float32)
            sq_sum = gl.sum(x * x, axis=0)
            rsqrt_val = gl.rsqrt(sq_sum / 512.0 + eps)
            x_normed = x * rsqrt_val
            pos = gl.load(position_ids_ptr + tok_idx)
            half_offs = gl.arange(0, 32, layout=ROPE_LAYOUT)
            cos = gl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
            sin = gl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)
            rope_even = 448 + half_offs * 2
            rope_odd = rope_even + 1
            x_re = gl.load(q_ptr + base + rope_even).to(gl.float32) * rsqrt_val
            x_ro = gl.load(q_ptr + base + rope_odd).to(gl.float32) * rsqrt_val
            q_out_e = x_re * cos - x_ro * sin
            q_out_o = x_re * sin + x_ro * cos
            nope_mask = offs < 448
            gl.store(q_ptr + base + offs, x_normed.to(gl.bfloat16), mask=nope_mask)
            gl.store(q_ptr + base + rope_even, q_out_e.to(gl.bfloat16))
            gl.store(q_ptr + base + rope_odd, q_out_o.to(gl.bfloat16))
        else:
            kv_idx = item_id - num_q_items
            slot_id = gl.load(slot_mapping_ptr + kv_idx)
            if slot_id >= 0:
                kv_base = kv_idx * stride_kv_tok
                offs = gl.arange(0, 512, layout=DATA_LAYOUT)
                kv_data = gl.load(kv_ptr + kv_base + offs)
                pos = gl.load(position_ids_ptr + kv_idx)
                half_offs = gl.arange(0, 32, layout=ROPE_LAYOUT)
                cos = gl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
                sin = gl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)
                rope_even = 448 + half_offs * 2
                rope_odd = rope_even + 1
                x_e = gl.load(kv_ptr + kv_base + rope_even).to(gl.float32)
                x_o = gl.load(kv_ptr + kv_base + rope_odd).to(gl.float32)
                out_e = x_e * cos - x_o * sin
                out_o = x_e * sin + x_o * cos
                block_idx = slot_id // CACHE_BLOCK_SIZE
                pos_in_block = slot_id % CACHE_BLOCK_SIZE
                cache_off = block_idx * stride_cache_block + pos_in_block * stride_cache_token
                nope_mask = offs < 448
                gl.store(k_cache_ptr + cache_off + offs, kv_data, mask=nope_mask)
                gl.store(k_cache_ptr + cache_off + rope_even, out_e.to(gl.bfloat16))
                gl.store(k_cache_ptr + cache_off + rope_odd, out_o.to(gl.bfloat16))
        item_id += gl.num_programs(0)


KERNEL = qnorm_rope_insert


def _grid(total_items: int) -> int:
    if total_items <= 516:
        return total_items
    if total_items <= 2193:
        return (3 * total_items + 3) // 4
    if total_items <= 8256:
        return (7 * total_items + 15) // 16
    return max(1, (total_items + 1) // 2 - 6144)


def fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache,
    eps=1e-6, cache_block_size=16,
):
    """In-place Q RMSNorm/RoPE and KV RoPE/cache insertion; return None.

    q [N,H,512], kv [N_insert,512], cache [blocks,block_size,512]: BF16.
    slot_mapping/position_ids: contiguous int64; cos_sin_cache [max_pos,64]: FP32.
    All tensors must be contiguous on one GPU. A negative slot skips insertion.
    The caller is responsible for valid positions, slots and non-overlapping
    insertions. This file supplies its own frozen grid and launch policy.
    """
    import torch
    if q.ndim != 3 or q.shape[-1] != 512 or q.shape[1] <= 0:
        raise ValueError("q must have shape [N,H,512], H > 0")
    if kv.ndim != 2 or kv.shape[-1] != 512:
        raise ValueError("kv must have shape [N_insert,512]")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (cache_block_size, 512) or cache_block_size <= 0:
        raise ValueError("k_cache must have shape [blocks,cache_block_size,512]")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != 64:
        raise ValueError("cos_sin_cache must have shape [max_pos,64]")
    if slot_mapping.ndim != 1 or position_ids.ndim != 1:
        raise ValueError("slot_mapping and position_ids must be vectors")
    if slot_mapping.numel() > kv.shape[0] or position_ids.numel() < max(q.shape[0], slot_mapping.numel()):
        raise ValueError("insufficient KV rows or position IDs")
    tensors = (q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache)
    dtypes = (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int64, torch.int64, torch.float32)
    if any(not t.is_cuda or t.device != q.device or not t.is_contiguous() or t.dtype != dtype
           for t, dtype in zip(tensors, dtypes)):
        raise ValueError("expected contiguous tensors with documented dtypes on one GPU")
    num_q_items = q.shape[0] * q.shape[1]
    total_items = num_q_items + slot_mapping.numel()
    if total_items == 0:
        return None
    KERNEL[(_grid(total_items),)](
        q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache,
        q.stride(0), q.stride(1), kv.stride(0), k_cache.stride(0), k_cache.stride(1),
        cos_sin_cache.stride(0), eps, num_q_items, total_items, q.shape[1],
        cache_block_size, **LAUNCH,
    )
    return None


def _self_test():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("a compatible HIP/DCU environment is required")
    torch.manual_seed(934518)
    tokens, heads, block_size = 17, 128, 16
    q = torch.randn((tokens, heads, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((tokens, 512), device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 3 + 2
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    slots[::3] = -1
    inv_freq = 1.0 / (10000.0 ** (torch.arange(32, device="cuda", dtype=torch.float32) / 32))
    phases = torch.outer(torch.arange(64, device="cuda", dtype=torch.float32), inv_freq)
    cos_sin = torch.cat((phases.cos(), phases.sin()), -1)
    cache = torch.full((3, block_size, 512), 0.25, device="cuda", dtype=torch.bfloat16)
    c, s = cos_sin[positions, :32], cos_sin[positions, 32:]
    qf = q.float()
    inv_rms = torch.rsqrt(qf.square().sum(-1, keepdim=True) / 512.0 + 1e-6)
    reference_q = (qf * inv_rms).to(torch.bfloat16)
    even, odd = qf[..., 448::2] * inv_rms, qf[..., 449::2] * inv_rms
    reference_q[..., 448::2] = (even * c[:, None, :] - odd * s[:, None, :]).to(torch.bfloat16)
    reference_q[..., 449::2] = (even * s[:, None, :] + odd * c[:, None, :]).to(torch.bfloat16)
    reference_kv = kv.clone()
    even, odd = kv[:, 448::2].float(), kv[:, 449::2].float()
    reference_kv[:, 448::2] = (even * c - odd * s).to(torch.bfloat16)
    reference_kv[:, 449::2] = (even * s + odd * c).to(torch.bfloat16)
    reference_cache = cache.clone()
    valid = slots >= 0
    reference_cache.view(-1, 512)[slots[valid]] = reference_kv[valid]
    fused_deepseek_v4_qnorm_rope_kv_rope_insert(q, kv, cache, slots, positions, cos_sin)
    torch.cuda.synchronize()
    torch.testing.assert_close(q, reference_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(cache, reference_cache, atol=2e-2, rtol=2e-2)
    variant = "baseline" if LAUNCH["num_warps"] == 2 else "optimized"
    print(f"PASS fused_qnorm_rope_insert {variant}: 17 tokens, 128 heads, grid={_grid(tokens * (heads + 1))}", flush=True)


if __name__ == "__main__":
    _self_test()
