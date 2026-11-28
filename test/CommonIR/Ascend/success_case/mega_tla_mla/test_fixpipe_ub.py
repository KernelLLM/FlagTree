"""Compare fixpipe to GM (proven) vs fixpipe to UB (failing)."""
import sys
sys.path.insert(0, "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut")
import torch, torch_npu
import triton.language as tl
import triton.language.extra.tla as tla
torch.npu.set_device(0)

# Proven: fixpipe to GM
@tla.jit
def fp_to_gm(Q_ptr, K_ptr, S_ptr,
              H: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    with tla.scope.cube():
        q_l1 = tla.alloc((D // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
        k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        l0c = tla.alloc((BN // 16, H // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(Q_ptr, (H, D)), q_l1)
        tla.load_gm_to_l1(tla.view(K_ptr, (BN, D)), k_l1)
        tla.cube.mmad(q_l1, k_l1, l0c, transpose_b=True, clear_accum=True)
        tla.store_l0c_to_gm(
            l0c, tla.view(S_ptr, (H, BN)),
            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))

# New: fixpipe to UB then vector stores to GM
FLAG0 = tl.constexpr(0)

@tla.jit
def fp_to_ub(Q_ptr, K_ptr, S_ptr,
             H: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    q_l1 = tla.alloc((D // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
    k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
    l0c = tla.alloc((BN // 16, H // 16, 16, 16), tl.float32, space="l0c")
    score = tla.alloc((H, BN), tl.float32, space="ub")
    with tla.scope.cube():
        tla.load_gm_to_l1(tla.view(Q_ptr, (H, D)), q_l1)
        tla.load_gm_to_l1(tla.view(K_ptr, (BN, D)), k_l1)
        tla.cube.mmad(q_l1, k_l1, l0c, transpose_b=True, clear_accum=True)
        tla.copy_l0c_to_ub(
            l0c, score,
            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, FLAG0)
    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG0)
        tla.store_ub_to_gm(score, tla.view(S_ptr, (H, BN)))

H, BN, D = 16, 64, 128
torch.manual_seed(42)
q = (torch.randn(H, D) * 0.1).to(torch.bfloat16).npu()
k = (torch.randn(BN, D) * 0.1).to(torch.bfloat16).npu()

s1 = torch.full((H, BN), float("nan"), dtype=torch.float32, device="npu")
fp_to_gm[(1,)](q, k, s1, H=H, BN=BN, D=D)
torch.npu.synchronize()

s2 = torch.full((H, BN), float("nan"), dtype=torch.float32, device="npu")
fp_to_ub[(1,)](q, k, s2, H=H, BN=BN, D=D)
torch.npu.synchronize()

ref = q.cpu().float() @ k.cpu().float().T
e1 = (s1.cpu() - ref).abs().max()
e2 = (s2.cpu() - ref).abs().max()
diff = (s1.cpu() - s2.cpu()).abs().max()
print(f"fp_to_gm vs ref:  max_err={e1:.6e}")
print(f"fp_to_ub vs ref:  max_err={e2:.6e}")
print(f"gm vs ub:         max_err={diff:.6e}")
