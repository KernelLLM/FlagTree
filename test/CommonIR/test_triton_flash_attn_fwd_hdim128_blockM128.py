"""
单元测试: triton_flash_attn_fwd_hdim128_blockM128.py
======================================================
与 PyTorch scaled dot-product attention 对比验证正确性。

运行:
    pytest test_triton_flash_attn_fwd_hdim128_blockM128.py -v
    python test_triton_flash_attn_fwd_hdim128_blockM128.py
"""

import pytest
import torch

try:
    from triton_flash_attn_fwd_hdim128_blockM128 import (
        flash_attn_fwd,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
    )
    HAS_KERNEL = True
    IMPORT_ERROR = ""
except Exception as e:
    HAS_KERNEL = False
    HEAD_DIM, BLOCK_M, BLOCK_N = 128, 128, 64
    IMPORT_ERROR = str(e)

requires_kernel = pytest.mark.skipif(not HAS_KERNEL, reason=f"Kernel import failed: {IMPORT_ERROR}")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# ---------------------------------------------------------------------------
# PyTorch fp32 参考实现
# ---------------------------------------------------------------------------


def ref_flash_attn(q, k, v, causal=False, scale=None):
    """
    纯 PyTorch 参考，接口与 flash_attn_fwd 完全相同。
      q, k, v : [B, seqlen, nheads, head_dim] fp16
      returns : (o: [B, M, H_q, D] fp16,  lse: [B, H_q, M] fp32)
    """
    B, M, H_q, D = q.shape
    _, N, H_k, _ = k.shape
    if scale is None:
        scale = D**-0.5
    h_ratio = H_q // H_k

    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float().repeat_interleave(h_ratio, dim=1)
    v_t = v.transpose(1, 2).float().repeat_interleave(h_ratio, dim=1)

    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale

    if causal:
        row_idx = torch.arange(M, device=q.device).unsqueeze(1)
        col_idx = torch.arange(N, device=q.device).unsqueeze(0)
        mask = (row_idx - (M - N)) >= col_idx  # 与 kernel 对齐
        scores = scores.masked_fill(~mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    o = torch.matmul(attn, v_t).to(torch.float16)
    return o.transpose(1, 2), lse


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def make_qkv(batch, seqlen_q, seqlen_k, nheads_q, nheads_k, head_dim=HEAD_DIM, seed=42, device="cuda"):
    torch.manual_seed(seed)
    q = torch.randn(batch, seqlen_q, nheads_q, head_dim, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(batch, seqlen_k, nheads_k, head_dim, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(batch, seqlen_k, nheads_k, head_dim, device=device, dtype=torch.float16) * 0.1
    return q, k, v


def assert_close(tri, ref, atol=1e-2, rtol=1e-2, name="tensor"):
    max_diff = (tri.float() - ref.float()).abs().max().item()
    assert torch.allclose(tri.float(), ref.float(), atol=atol,
                          rtol=rtol), (f"[{name}] max_diff={max_diff:.6f} > atol={atol}, rtol={rtol}")


# ---------------------------------------------------------------------------
# 1. 输出基础属性
# ---------------------------------------------------------------------------


@requires_cuda
@requires_kernel
class TestOutputProperties:

    def test_shape(self):
        B, M, N, H, D = 2, 128, 128, 4, HEAD_DIM
        q, k, v = make_qkv(B, M, N, H, H)
        o, lse = flash_attn_fwd(q, k, v)
        assert o.shape == (B, M, H, D)
        assert lse.shape == (B, H, M)

    def test_dtype(self):
        q, k, v = make_qkv(1, 128, 128, 1, 1)
        o, lse = flash_attn_fwd(q, k, v)
        assert o.dtype == torch.float16
        assert lse.dtype == torch.float32

    def test_device(self):
        q, k, v = make_qkv(1, 128, 128, 1, 1)
        o, lse = flash_attn_fwd(q, k, v)
        assert o.device == q.device
        assert lse.device == q.device

    def test_no_nan_inf(self):
        q, k, v = make_qkv(2, 256, 256, 2, 2)
        o, lse = flash_attn_fwd(q, k, v)
        assert not torch.isnan(o).any(), "o 含 NaN"
        assert not torch.isinf(o).any(), "o 含 Inf"
        assert not torch.isnan(lse).any(), "lse 含 NaN"

    def test_scale_default_equals_explicit(self):
        q, k, v = make_qkv(1, 128, 128, 1, 1)
        o1, l1 = flash_attn_fwd(q, k, v, scale=None)
        o2, l2 = flash_attn_fwd(q, k, v, scale=HEAD_DIM**-0.5)
        assert_close(o1, o2, name="o")
        assert_close(l1, l2, name="lse")


# ---------------------------------------------------------------------------
# 2. 非因果精度测试
# ---------------------------------------------------------------------------


@requires_cuda
@requires_kernel
class TestNonCausalAccuracy:

    ATOL, RTOL = 1e-2, 1e-2

    @pytest.mark.parametrize("seqlen", [128, 256, 512])
    def test_square(self, seqlen):
        q, k, v = make_qkv(1, seqlen, seqlen, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=False)
        o_r, l_r = ref_flash_attn(q, k, v, causal=False)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, f"o[{seqlen}]")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, f"lse[{seqlen}]")

    def test_batch2_heads4(self):
        q, k, v = make_qkv(2, 256, 256, 4, 4)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=False)
        o_r, l_r = ref_flash_attn(q, k, v, causal=False)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse")

    def test_gqa_ratio2(self):
        """GQA: H_q=4, H_k=2 (h_ratio=2)。"""
        q, k, v = make_qkv(1, 128, 128, 4, 2)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=False)
        o_r, l_r = ref_flash_attn(q, k, v, causal=False)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o_gqa")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse_gqa")

    def test_custom_scale(self):
        scale = 0.05
        q, k, v = make_qkv(1, 128, 128, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=False, scale=scale)
        o_r, l_r = ref_flash_attn(q, k, v, causal=False, scale=scale)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse")


# ---------------------------------------------------------------------------
# 3. 因果注意力精度测试
# ---------------------------------------------------------------------------


@requires_cuda
@requires_kernel
class TestCausalAccuracy:

    ATOL, RTOL = 1e-2, 1e-2

    @pytest.mark.parametrize("seqlen", [128, 256, 512])
    def test_causal_square(self, seqlen):
        """因果，M == N。"""
        q, k, v = make_qkv(1, seqlen, seqlen, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=True)
        o_r, l_r = ref_flash_attn(q, k, v, causal=True)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, f"o_causal[{seqlen}]")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, f"lse_causal[{seqlen}]")

    def test_causal_batch2_heads4(self):
        q, k, v = make_qkv(2, 256, 256, 4, 4)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=True)
        o_r, l_r = ref_flash_attn(q, k, v, causal=True)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse")

    def test_causal_gqa(self):
        """因果 + GQA。"""
        q, k, v = make_qkv(1, 128, 128, 4, 2)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=True)
        o_r, l_r = ref_flash_attn(q, k, v, causal=True)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o_causal_gqa")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse_causal_gqa")

    def test_causal_first_row_attends_only_self(self):
        """因果 + M==N: 第 0 行只能 attend 位置 0，输出应等价于 V[0]（忽略 scale 对 softmax 的影响）。"""
        M = N = 128
        q, k, v = make_qkv(1, M, N, 1, 1)
        o_t, _ = flash_attn_fwd(q, k, v, causal=True)
        # 第 0 个 query 仅 attend key[0]，所以 o[0] ≈ v[0]
        # 用参考实现验证（不直接检验 v，因为 scale 不确定）
        o_r, _ = ref_flash_attn(q, k, v, causal=True)
        assert_close(o_t[0, 0], o_r[0, 0], self.ATOL, self.RTOL, "o[0,0]")


# ---------------------------------------------------------------------------
# 4. 边界序列长度测试
# ---------------------------------------------------------------------------


@requires_cuda
@requires_kernel
class TestEdgeCases:

    ATOL, RTOL = 1e-2, 1e-2

    def test_seqlen_exactly_block_m(self):
        """seqlen_q == seqlen_k == BLOCK_M，恰好一个 M-block。"""
        q, k, v = make_qkv(1, BLOCK_M, BLOCK_M, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v)
        o_r, l_r = ref_flash_attn(q, k, v)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse")

    def test_seqlen_two_block_m(self):
        """seqlen == 2 * BLOCK_M，两个 M-block。"""
        q, k, v = make_qkv(1, BLOCK_M * 2, BLOCK_M * 2, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v)
        o_r, l_r = ref_flash_attn(q, k, v)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse")

    def test_seqlen_non_multiple_of_block(self):
        """seqlen_q 和 seqlen_k 非 BLOCK_M / BLOCK_N 整数倍（边界 mask 路径）。"""
        q, k, v = make_qkv(1, 192, 160, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v)
        o_r, l_r = ref_flash_attn(q, k, v)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o_nonmul")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse_nonmul")

    def test_seqlen_q_longer_than_k(self):
        """M > N（decoder cross-attention 场景）。"""
        q, k, v = make_qkv(1, 256, 128, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=False)
        o_r, l_r = ref_flash_attn(q, k, v, causal=False)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse")

    def test_seqlen_k_longer_than_q(self):
        """N > M（encoder cross-attention 场景）。"""
        q, k, v = make_qkv(1, 128, 256, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=False)
        o_r, l_r = ref_flash_attn(q, k, v, causal=False)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse")

    def test_causal_seqlen_k_longer_than_q(self):
        """因果 + N > M（KV cache 场景：K 包含历史 token）。"""
        q, k, v = make_qkv(1, 128, 256, 1, 1)
        o_t, l_t = flash_attn_fwd(q, k, v, causal=True)
        o_r, l_r = ref_flash_attn(q, k, v, causal=True)
        assert_close(o_t, o_r, self.ATOL, self.RTOL, "o_kvcache")
        assert_close(l_t, l_r, self.ATOL, self.RTOL, "lse_kvcache")


# ---------------------------------------------------------------------------
# 5. 数值一致性测试（确定性 / 独立性）
# ---------------------------------------------------------------------------


@requires_cuda
@requires_kernel
class TestNumericalConsistency:

    def test_deterministic(self):
        """相同输入多次运行结果相同（无随机性）。"""
        q, k, v = make_qkv(1, 256, 256, 2, 2)
        o1, l1 = flash_attn_fwd(q, k, v)
        o2, l2 = flash_attn_fwd(q, k, v)
        assert torch.equal(o1, o2), "输出不确定"
        assert torch.equal(l1, l2), "lse 不确定"

    def test_batch_independence(self):
        """batch 内各样本互不影响：拼成一个 batch 与单独计算结果相同。"""
        q0, k0, v0 = make_qkv(1, 256, 256, 1, 1, seed=0)
        q1, k1, v1 = make_qkv(1, 256, 256, 1, 1, seed=1)

        o0_single, l0_single = flash_attn_fwd(q0, k0, v0)
        o1_single, l1_single = flash_attn_fwd(q1, k1, v1)

        q_batch = torch.cat([q0, q1], dim=0)
        k_batch = torch.cat([k0, k1], dim=0)
        v_batch = torch.cat([v0, v1], dim=0)
        o_batch, l_batch = flash_attn_fwd(q_batch, k_batch, v_batch)

        assert torch.equal(o_batch[0:1], o0_single), "batch[0] 与单独计算不一致"
        assert torch.equal(o_batch[1:2], o1_single), "batch[1] 与单独计算不一致"
        assert torch.equal(l_batch[0:1], l0_single), "lse batch[0] 不一致"
        assert torch.equal(l_batch[1:2], l1_single), "lse batch[1] 不一致"

    def test_zero_query_output_near_zero(self):
        """Q 全零时 softmax 均匀分布，o ≈ mean(V) per row（近似检验）。"""
        B, M, N, H = 1, 128, 128, 1
        q = torch.zeros(B, M, H, HEAD_DIM, device="cuda", dtype=torch.float16)
        k, v = make_qkv(1, M, N, H, H)[1:]  # 只要 k, v
        o_t, _ = flash_attn_fwd(q, k, v, causal=False)
        o_r, _ = ref_flash_attn(q, k, v, causal=False)
        assert_close(o_t, o_r, atol=1e-2, rtol=1e-2, name="o_zeroq")


# ---------------------------------------------------------------------------
# 6. LSE 正确性专项测试
# ---------------------------------------------------------------------------


@requires_cuda
@requires_kernel
class TestLSECorrectness:

    def test_lse_matches_pytorch_logsumexp(self):
        """LSE 与 PyTorch logsumexp(Q@K^T * scale, dim=-1) 对比。"""
        q, k, v = make_qkv(1, 128, 128, 1, 1)
        scale = HEAD_DIM**-0.5
        _, lse_t = flash_attn_fwd(q, k, v, causal=False, scale=scale)

        q_t = q.transpose(1, 2).float()
        k_t = k.transpose(1, 2).float()
        scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale
        lse_r = torch.logsumexp(scores, dim=-1)  # [B, H, M]

        assert_close(lse_t, lse_r, atol=1e-2, rtol=1e-2, name="lse")

    def test_lse_causal_vs_non_causal_differ(self):
        """因果与非因果的 LSE 在非末尾行应有差异。"""
        q, k, v = make_qkv(1, 128, 128, 1, 1)
        _, lse_nc = flash_attn_fwd(q, k, v, causal=False)
        _, lse_c = flash_attn_fwd(q, k, v, causal=True)
        # 最后一行（行索引 M-1）两者相同（因果最后一行 attend 全部 K）
        # 第 0 行应不同（非因果 attend 全部 K，因果只 attend K[0]）
        assert not torch.allclose(lse_nc[0, 0, 0], lse_c[0, 0, 0]), \
            "因果与非因果 lse[0] 不应相同"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
