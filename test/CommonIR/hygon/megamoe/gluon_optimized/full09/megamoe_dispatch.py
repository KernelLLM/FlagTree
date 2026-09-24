"""Validated stage-pair dispatch for the MegaMoE Gluon B0/H1 kernels.

H1 is selected only for the six production signatures measured in the final
protocol. Every other supported input uses the exact-layout B0 Gluon pair.
"""

from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import torch
import triton

from flag_gems.runtime import torch_device_fn


ROOT = Path(__file__).resolve().parents[1]
RAW_RUN = triton.runtime.JITFunction.run
SOURCES = {
    "equivalent": (
        ROOT
        / "gluon_baseline/mixtral-m16-h4096-i14336-e8-k2-ue8m0-_fp8_fp4_mega_moe_l1_kernel-5cb970117bbb46ee.py",
        ROOT
        / "gluon_baseline/mixtral-m16-h4096-i14336-e8-k2-ue8m0-_fp8_fp4_mega_moe_l2_kernel-590137d8f2e23261.py",
    ),
    "h1": (
        ROOT / "gluon_optimized/megamoe_h1_l1_reduction_first.py",
        ROOT / "gluon_optimized/megamoe_h1_l2_reduction_first.py",
    ),
}
SOURCE_SHA256 = {
    "equivalent": (
        "12a38c392e5a8db6eac285e61b3178cc5cc8b37b8dc7b14b19a71939df1549dc",
        "64ead53c0a8725cf1894aa160c209fe49fdaa88e14c9d6c8e21d2642def3cb2a",
    ),
    "h1": (
        "eac9ec730d624f66ceaba7aa3cf80bd4d7ce43d9932a4a2cd57360d7c2daf81f",
        "5d0277c316962b0c956d38c9175f5b9f9f2187d6693669aa50958bd0eb76e1ba",
    ),
}
OPTIMIZED_SIGNATURES = {
    (m, h, i, e, topk)
    for h, i, e, topk in ((4096, 14336, 8, 2), (7168, 2048, 256, 8))
    for m in (1, 4, 16)
}


def _check_supported_scales(scale: torch.Tensor, name: str) -> bool:
    if scale.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if scale.dtype == torch.uint8:
        return True
    raise TypeError(f"{name} must be a floating scale tensor or uint8 UE8M0 scales")


def _load(path: Path, expected_sha256: str):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(("reviewed Gluon source hash mismatch", str(path), actual, expected_sha256))
    module_name = "_megamoe_dispatch_" + actual[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.KERNEL


@lru_cache(maxsize=2)
def _kernels(implementation: str):
    return tuple(
        _load(path, expected)
        for path, expected in zip(SOURCES[implementation], SOURCE_SHA256[implementation])
    )


def implementation_for(
    x_fp8: torch.Tensor,
    topk_idx: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_scales: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_scales: torch.Tensor,
    activation_clamp: float | None,
) -> str:
    m, h = x_fp8.shape
    topk = topk_idx.shape[1]
    e = l1_weights.shape[0]
    i = l2_weights.shape[2] * 2
    scale_is_ue8m0 = _check_supported_scales(l1_scales, "l1_scales")
    if _check_supported_scales(l2_scales, "l2_scales") != scale_is_ue8m0:
        raise TypeError("l1_scales and l2_scales must use the same scale representation")
    signature = (m, h, i, e, topk)
    if scale_is_ue8m0 and activation_clamp is None and signature in OPTIMIZED_SIGNATURES:
        return "h1"
    return "equivalent"


def fp8_fp4_mega_moe(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_scales: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    """Run reviewed H1 for final production signatures, otherwise exact B0."""
    if x_fp8.ndim != 2:
        raise ValueError("x_fp8 must be [num_tokens, hidden]")
    if topk_idx.shape != topk_weights.shape:
        raise ValueError("topk_idx and topk_weights must have the same shape")
    if topk_idx.shape[0] != x_fp8.shape[0]:
        raise ValueError("routing tensors must have the same token count as x_fp8")
    if l1_weights.ndim != 3 or l2_weights.ndim != 3:
        raise ValueError("l1_weights and l2_weights must be 3D packed FP4 tensors")

    num_tokens, hidden = x_fp8.shape
    top_k = topk_idx.shape[1]
    num_experts, l1_n, l1_k_half = l1_weights.shape
    l2_experts, l2_h, l2_i_half = l2_weights.shape
    intermediate = l2_i_half * 2
    if l2_experts != num_experts or l2_h != hidden:
        raise ValueError("l2 weight shape is inconsistent with x/l1 weights")
    if l1_n != 2 * intermediate or l1_k_half * 2 != hidden:
        raise ValueError("l1 weight shape must be [E, 2 * I, H // 2]")
    if hidden % 32 != 0 or intermediate % 32 != 0:
        raise ValueError("hidden and intermediate must be multiples of 32")
    if x_scale.shape != (num_tokens, hidden // 32):
        raise ValueError("x_scale must be [num_tokens, hidden // 32]")
    if top_k > 8:
        raise ValueError("this Gluon implementation supports top_k <= 8")

    implementation = implementation_for(
        x_fp8,
        topk_idx,
        l1_weights,
        l1_scales,
        l2_weights,
        l2_scales,
        activation_clamp,
    )
    scale_is_ue8m0 = l1_scales.dtype == torch.uint8
    l1_kernel, l2_kernel = _kernels(implementation)
    if out is None:
        out = torch.empty(
            (num_tokens, hidden), device=x_fp8.device, dtype=torch.bfloat16
        )
    if out.shape != (num_tokens, hidden):
        raise ValueError("out must be [num_tokens, hidden]")
    l1_out = torch.empty(
        (num_tokens, top_k, 2 * intermediate),
        device=x_fp8.device,
        dtype=torch.float32,
    )

    l1_grid = (num_tokens, top_k, triton.cdiv(2 * intermediate, 16))
    l2_grid = (num_tokens, triton.cdiv(hidden, 16))
    with torch_device_fn.device(x_fp8.device):
        RAW_RUN(
            l1_kernel,
            x_fp8,
            x_scale,
            topk_idx,
            l1_weights,
            l1_scales,
            l1_out,
            num_tokens,
            hidden,
            intermediate,
            top_k,
            x_fp8.stride(0),
            x_fp8.stride(1),
            x_scale.stride(0),
            x_scale.stride(1),
            topk_idx.stride(0),
            topk_idx.stride(1),
            l1_weights.stride(0),
            l1_weights.stride(1),
            l1_weights.stride(2),
            l1_scales.stride(0),
            l1_scales.stride(1),
            l1_scales.stride(2),
            l1_out.stride(0),
            l1_out.stride(1),
            l1_out.stride(2),
            grid=l1_grid,
            warmup=False,
            BLOCK_N=16,
            BLOCK_K=32,
            SCALE_IS_UE8M0=scale_is_ue8m0,
        )
        RAW_RUN(
            l2_kernel,
            topk_idx,
            topk_weights,
            l1_out,
            l2_weights,
            l2_scales,
            out,
            num_tokens,
            hidden,
            intermediate,
            top_k,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            l1_out.stride(0),
            l1_out.stride(1),
            l1_out.stride(2),
            l2_weights.stride(0),
            l2_weights.stride(1),
            l2_weights.stride(2),
            l2_scales.stride(0),
            l2_scales.stride(1),
            l2_scales.stride(2),
            out.stride(0),
            out.stride(1),
            grid=l2_grid,
            warmup=False,
            BLOCK_H=16,
            BLOCK_I=32,
            SCALE_IS_UE8M0=scale_is_ue8m0,
            ACTIVATION_CLAMP=(
                -1.0 if activation_clamp is None else float(activation_clamp)
            ),
        )
    return out
