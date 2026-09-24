"""Shape-aware Gluon layout dispatch selected from the gfx936 sweeps.

This module keeps launch choice explicit so the measured kernel is reproducible.
The returned tuple is ``(kernel, launch_kwargs, layout_id)``.
"""

import torch

from ..gluon_baseline.rms_norm_grad_dx_kernel_99b74d3bae_248f54e5234a import (
    KERNEL as DX_BASELINE,
)
from ..gluon_baseline.rms_norm_kernel_0aaa02d6c8_4a682d6d42c7 import (
    KERNEL as FORWARD_BASELINE,
)
from ..gluon_baseline.rms_norm_loop_t1024_w8 import KERNEL as LOOP_BASE_T1024_W8
from ..gluon_baseline.rms_norm_loop_t1024_w16 import KERNEL as LOOP_BASE_T1024_W16
from ..gluon_baseline.rms_norm_loop_t8192_w8 import KERNEL as LOOP_BASE_T8192_W8
from .rms_norm_grad_dw_w2_r1_c4 import KERNEL as DW_W2_R1_C4
from .rms_norm_grad_dw_w4_r1_c2 import KERNEL as DW_W4_R1_C2
from .rms_norm_grad_dx_w1_s1 import KERNEL as DX_W1_S1
from .rms_norm_grad_dx_w1_s8 import KERNEL as DX_W1_S8
from .rms_norm_grad_dx_w4_s2 import KERNEL as DX_W4_S2
from .rms_norm_grad_dx_w4_s4 import KERNEL as DX_W4_S4
from .rms_norm_grad_dx_w8_s2 import KERNEL as DX_W8_S2
from .rms_norm_grad_dx_w8_s4 import KERNEL as DX_W8_S4
from .rms_norm_loop_t8192_w16_unified import KERNEL as LOOP_UNIFIED_T8192_W16
from .rms_norm_loop_t8192_w8_unified import KERNEL as LOOP_UNIFIED_T8192_W8
from .rms_norm_w1_s1 import KERNEL as FORWARD_W1_S1
from .rms_norm_w1_s8 import KERNEL as FORWARD_W1_S8
from .rms_norm_w4_s2 import KERNEL as FORWARD_W4_S2
from .rms_norm_w4_s4 import KERNEL as FORWARD_W4_S4
from .rms_norm_w8_s2 import KERNEL as FORWARD_W8_S2


def _launch(num_warps, waves_per_eu=1):
    return {
        "num_warps": num_warps,
        "num_stages": 2,
        "waves_per_eu": waves_per_eu,
    }


def forward_spec(m, n, dtype):
    """Choose the small-N data/reduction/broadcast layout."""
    block = 1 << (n - 1).bit_length()
    if dtype == torch.bfloat16:
        if block <= 128:
            return FORWARD_W1_S1, _launch(1), "w1_s1"
        if block <= 512:
            if m >= 32:
                return FORWARD_W1_S8, _launch(1), "w1_s8"
            return FORWARD_BASELINE, _launch(4), "baseline_w4_s1"
        if block <= 1024:
            if m < 128:
                return FORWARD_W8_S2, _launch(8), "w8_s2"
            return FORWARD_W1_S8, _launch(1), "w1_s8"
        if block <= 2048:
            if m <= 1:
                return FORWARD_W8_S2, _launch(8), "w8_s2"
            return FORWARD_W4_S2, _launch(4), "w4_s2"
        if block <= 4096:
            if m <= 1:
                return FORWARD_W8_S2, _launch(8), "w8_s2"
            return FORWARD_W4_S4, _launch(4), "w4_s4"
    else:
        if block <= 128:
            return FORWARD_W1_S1, _launch(1), "w1_s1"
        if block <= 512:
            return FORWARD_W1_S1, _launch(1), "w1_s1"
        if block <= 4096:
            return FORWARD_W4_S4, _launch(4), "w4_s4"
    raise ValueError(f"small-N dispatch does not cover N={n}")


def dx_spec(m, n, dtype):
    """Choose the backward-dx reduction/broadcast layout."""
    block = 1 << (n - 1).bit_length()
    if dtype == torch.bfloat16:
        if block <= 128:
            return DX_W1_S1, _launch(1), "w1_s1"
        if block <= 512:
            return DX_W4_S2, _launch(4), "w4_s2"
        if block <= 1024:
            return DX_W1_S8, _launch(1), "w1_s8"
        if block <= 2048:
            return DX_BASELINE, _launch(4), "baseline_w4_s1"
        if block <= 4096:
            return DX_W8_S2, _launch(8), "w8_s2"
    else:
        if block <= 32:
            return DX_BASELINE, _launch(4), "baseline_w4_s1"
        if block <= 512:
            return DX_BASELINE, _launch(4), "baseline_w4_s1"
        if block <= 1024:
            return DX_W4_S4, _launch(4), "w4_s4"
        if block <= 4096:
            return DX_W8_S4, _launch(8), "w8_s4"
    raise ValueError(f"dx dispatch does not cover N={n}")


def dw_spec(dtype):
    """Use the reduction result's SliceLayout directly at the store."""
    if dtype == torch.bfloat16:
        return DW_W4_R1_C2, _launch(4), "w4_r1_c2"
    return DW_W2_R1_C4, _launch(2), "w2_r1_c4"


def loop_baseline_spec(n, dtype):
    """Equivalent Gluon specialization matching the observed TLE autotuner."""
    if n <= 16384:
        return LOOP_BASE_T8192_W8, _launch(8), "t8192_w8"
    if dtype == torch.bfloat16:
        return LOOP_BASE_T1024_W8, _launch(8), "t1024_w8"
    return LOOP_BASE_T1024_W16, _launch(16), "t1024_w16"


def loop_optimized_spec(m, n, dtype):
    """Choose a unified layout only in ranges where it paid for conversion removal."""
    if n <= 8192:
        return LOOP_UNIFIED_T8192_W8, _launch(8), "t8192_w8_unified"
    if n <= 16384:
        return LOOP_UNIFIED_T8192_W16, _launch(16), "t8192_w16_unified"
    if dtype == torch.float32:
        return LOOP_BASE_T1024_W16, _launch(16), "t1024_w16_baseline_retained"
    return LOOP_BASE_T1024_W8, _launch(8), "t1024_w8_baseline_retained"
