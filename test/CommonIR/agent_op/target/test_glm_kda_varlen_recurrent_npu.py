"""Numerical comparison: optimized v0 kernel vs upstream sglang reference kernel.

Runs both kernels on identical inputs and verifies output/state match at
plan tolerances. Uses the original kda_recurrent_npu.py from sglang as oracle.

Production dispatch uses BV=64 reference kernel (bitwise-identical to upstream).
An experimental fused path (BV<64) is gated behind GLM53_KDA_ENABLE_FUSED=1.

Plan tolerances:
  - Output: rtol=0, atol=0 (bitwise match for BV=64)
  - State: rtol=2e-4, atol=2e-5 (FP32 accumulation tolerance)
"""

import os
import sys
import torch
import torch_npu

# Explicitly control fused execution: production mode (disabled)
os.environ.setdefault("GLM53_KDA_ENABLE_FUSED", "0")

sys.path.insert(0, '/public-flash/yuansheng/op/flagos/FlagTree/test/CommonIR/agent_op/target')
sys.path.insert(0, '/public-flash/yuansheng/models/sglang/python/sglang/srt/hardware_backend/npu/attention/glm53')

from glm_kda_varlen_recurrent_npu_v0 import glm_kda_varlen_recurrent_npu as optimized_fn
from kda_recurrent_npu import glm_kda_varlen_recurrent_npu as reference_fn


def offsets(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return torch.tensor(result, dtype=torch.int32, device="npu:0")


def test_numerical_match(seq_lengths, heads, rtol=0, atol=0):
    """Compare optimized vs reference kernel outputs.

    With fused path disabled (default), both kernels use BV=64 and should
    match bitwise (rtol=0, atol=0). With fused path enabled, BV<64 introduces
    accumulation-order differences (~1e-2 max).
    """
    torch.manual_seed(42)
    total = sum(seq_lengths)
    num_seqs = len(seq_lengths)
    initial = torch.randn(num_seqs + 8, heads, 128, 128, device="npu:0", dtype=torch.float32) * 0.1
    initial = initial.transpose(-1, -2)  # non-contiguous, stride-based addressing
    slots = torch.arange(1, num_seqs + 1, dtype=torch.int64, device="npu:0")
    starts = offsets(seq_lengths)

    q = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    k = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    v = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    a = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    b = torch.randn(1, total, heads, device="npu:0").bfloat16()

    A_log = torch.randn(heads, device="npu:0") * 0.1
    dt_bias = torch.randn(heads * 128, device="npu:0") * 0.1
    lower_bound = -5.0

    init_ref = initial.clone()
    init_opt = initial.clone()

    out_ref = reference_fn(
        A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
        q=q, k=k, v=v, a=a, b=b,
        initial_state_source=init_ref,
        initial_state_indices=slots, cu_seqlens=starts, prefill=True,
    )
    out_opt = optimized_fn(
        A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
        q=q, k=k, v=v, a=a, b=b,
        initial_state_source=init_opt,
        initial_state_indices=slots, cu_seqlens=starts, prefill=True,
    )

    assert out_ref.shape == out_opt.shape, \
        f"Shape mismatch: ref={out_ref.shape} opt={out_opt.shape}"

    max_diff = (out_ref.float() - out_opt.float()).abs().max().item()
    rel_diff = ((out_ref.float() - out_opt.float()).abs() /
                (out_ref.float().abs() + 1e-8)).max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_ref.float().flatten().unsqueeze(0),
        out_opt.float().flatten().unsqueeze(0),
    ).item()

    match = torch.allclose(out_ref.float(), out_opt.float(), rtol=rtol, atol=atol)
    return match, max_diff, rel_diff, cos_sim


def test_state_update(heads=4):
    """Verify that both kernels update the recurrent state consistently."""
    torch.manual_seed(42)
    total = 128
    initial = torch.randn(12, heads, 128, 128, device="npu:0", dtype=torch.float32) * 0.1
    initial = initial.transpose(-1, -2)
    slots = torch.tensor([1], dtype=torch.int64, device="npu:0")
    starts = offsets([total])

    q = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    k = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    v = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    a = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    b = torch.randn(1, total, heads, device="npu:0").bfloat16()
    A_log = torch.randn(heads, device="npu:0") * 0.1
    dt_bias = torch.randn(heads * 128, device="npu:0") * 0.1

    init_ref = initial.clone()
    init_opt = initial.clone()

    reference_fn(
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        q=q, k=k, v=v, a=a, b=b,
        initial_state_source=init_ref,
        initial_state_indices=slots, cu_seqlens=starts, prefill=True,
    )
    optimized_fn(
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        q=q, k=k, v=v, a=a, b=b,
        initial_state_source=init_opt,
        initial_state_indices=slots, cu_seqlens=starts, prefill=True,
    )

    # Compare state at slot 1 (both kernels write back)
    state_ref = init_ref[1].float()
    state_opt = init_opt[1].float()
    max_diff = (state_ref - state_opt).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        state_ref.flatten().unsqueeze(0),
        state_opt.flatten().unsqueeze(0),
    ).item()
    # Production BV=64 path should match bitwise (diff = 0).
    # Fused BV<64 path accumulates differently (~rtol=2e-4, atol=2e-5).
    match = torch.allclose(state_ref, state_opt, rtol=2e-4, atol=2e-5)
    return match, max_diff, cos_sim


def test_tracking(heads=4):
    """Verify state tracking (prefix checkpoint) works in optimized kernel."""
    torch.manual_seed(42)
    total = 256
    initial = torch.randn(12, heads, 128, 128, device="npu:0", dtype=torch.float32) * 0.1
    initial = initial.transpose(-1, -2)
    slots = torch.tensor([1], dtype=torch.int64, device="npu:0")
    starts = offsets([total])

    q = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    k = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    v = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    a = (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
    b = torch.randn(1, total, heads, device="npu:0").bfloat16()
    A_log = torch.randn(heads, device="npu:0") * 0.1
    dt_bias = torch.randn(heads * 128, device="npu:0") * 0.1

    # Track state at token 128
    track_slots = torch.tensor([2], dtype=torch.int64, device="npu:0")
    track_lens = torch.tensor([128], dtype=torch.int64, device="npu:0")

    init_ref = initial.clone()
    init_opt = initial.clone()

    reference_fn(
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        q=q, k=k, v=v, a=a, b=b,
        initial_state_source=init_ref,
        initial_state_indices=slots, cu_seqlens=starts, prefill=True,
        track_state_indices=track_slots, track_state_lens=track_lens,
    )
    optimized_fn(
        A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
        q=q, k=k, v=v, a=a, b=b,
        initial_state_source=init_opt,
        initial_state_indices=slots, cu_seqlens=starts, prefill=True,
        track_state_indices=track_slots, track_state_lens=track_lens,
    )

    # Compare tracked state at slot 2
    state_ref = init_ref[2].float()
    state_opt = init_opt[2].float()
    max_diff = (state_ref - state_opt).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        state_ref.flatten().unsqueeze(0),
        state_opt.flatten().unsqueeze(0),
    ).item()
    match = torch.allclose(state_ref, state_opt, rtol=2e-4, atol=2e-5)
    return match, max_diff, cos_sim


def test_output_determinism(seq_len=256, heads=4, runs=5):
    """Run the optimized kernel multiple times and check output is deterministic."""
    torch.manual_seed(42)
    initial = torch.randn(8, heads, 128, 128, device="npu:0", dtype=torch.float32) * 0.1
    initial = initial.transpose(-1, -2)
    slots = torch.tensor([1], dtype=torch.int64, device="npu:0")
    starts = offsets([seq_len])

    q = (torch.randn(1, seq_len, heads, 128, device="npu:0") * 0.2).bfloat16()
    k = (torch.randn(1, seq_len, heads, 128, device="npu:0") * 0.2).bfloat16()
    v = (torch.randn(1, seq_len, heads, 128, device="npu:0") * 0.2).bfloat16()
    a = (torch.randn(1, seq_len, heads, 128, device="npu:0") * 0.2).bfloat16()
    b = torch.randn(1, seq_len, heads, device="npu:0").bfloat16()
    A_log = torch.randn(heads, device="npu:0") * 0.1
    dt_bias = torch.randn(heads * 128, device="npu:0") * 0.1

    outputs = []
    for _ in range(runs):
        state = initial.clone()
        out = optimized_fn(
            A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
            q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state,
            initial_state_indices=slots, cu_seqlens=starts, prefill=True,
        )
        outputs.append(out.clone())

    # All runs should produce identical output
    max_diff = 0.0
    for i in range(1, len(outputs)):
        diff = (outputs[0].float() - outputs[i].float()).abs().max().item()
        max_diff = max(max_diff, diff)

    return max_diff == 0.0, max_diff


if __name__ == "__main__":
    torch.npu.set_device(0)

    all_pass = True

    # --- Output comparison ---
    print("=== Output comparison (optimized v0 vs reference) ===")
    production_configs = [
        ([129], 4, "single seq T=129, H=4"),
        ([256], 4, "single seq T=256, H=4"),
        ([512], 4, "single seq T=512, H=4"),
        ([1024], 4, "single seq T=1024, H=4"),
        ([256], 8, "single seq T=256, H=8"),
        ([512], 8, "single seq T=512, H=8"),
        ([129, 257], 4, "varlen 2-seq, H=4"),
        ([64, 128, 256], 4, "varlen 3-seq, H=4"),
        ([3264], 32, "single seq T=3264, H=32"),
    ]

    for lengths, heads, desc in production_configs:
        match, max_diff, rel_diff, cos_sim = test_numerical_match(lengths, heads)
        status = "✓" if match else "✗"
        print(f"  {status} {desc}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}, cos_sim={cos_sim:.8f}")
        if not match:
            all_pass = False

    # --- State update consistency ---
    print("\n=== State update consistency ===")
    for heads in [4, 8, 32]:
        match, max_diff, cos_sim = test_state_update(heads)
        status = "✓" if match else "✗"
        print(f"  {status} state update H={heads}: max_diff={max_diff:.2e}, cos_sim={cos_sim:.8f}")
        if not match:
            all_pass = False

    # --- State tracking ---
    print("\n=== State tracking (prefix checkpoint) ===")
    for heads in [4, 8]:
        match, max_diff, cos_sim = test_tracking(heads)
        status = "✓" if match else "✗"
        print(f"  {status} state tracking H={heads}: max_diff={max_diff:.2e}, cos_sim={cos_sim:.8f}")
        if not match:
            all_pass = False

    # --- Determinism ---
    print("\n=== Output determinism (5 runs) ===")
    for seq_len, heads in [(256, 4), (512, 8)]:
        match, max_diff = test_output_determinism(seq_len, heads)
        status = "✓" if match else "✗"
        print(f"  {status} determinism T={seq_len}, H={heads}: max_diff={max_diff:.2e}")
        if not match:
            all_pass = False

    # --- Final verdict ---
    if all_pass:
        print("\n✅ All accuracy tests passed")
        exit(0)
    else:
        print("\n❌ Some accuracy tests failed")
        exit(1)
