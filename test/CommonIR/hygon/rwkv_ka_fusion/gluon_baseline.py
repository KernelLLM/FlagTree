import logging
'Historical H=8/N=64 RWKV calculation expressed directly in public Gluon.'
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3

@g.jit
def rwkv_ka_fusion_kernel(k_ptr, kk_ptr, a_ptr, ka_ptr, o_k_ptr, o_kk_ptr, o_kka_ptr, T, C, H, N, N_size: gl.constexpr, block_size: gl.constexpr):
    pid = gl.program_id(0)
    k_start = pid * block_size
    for i in range(0, H):
        lane = gl.arange(0, N_size, layout=gl.BlockedLayout([1], [64], [4], [0]))
        offs = k_start + i * N + lane
        k = cdna3.buffer_load(k_ptr, offs.to(gl.int32), offs < T * C, 0.0)
        a = cdna3.buffer_load(a_ptr, offs.to(gl.int32), offs < T * C, 0.0)
        c_offs = i * N + lane
        ka = cdna3.buffer_load(ka_ptr, c_offs.to(gl.int32), c_offs < C, 0.0)
        kk = cdna3.buffer_load(kk_ptr, c_offs.to(gl.int32), c_offs < C, 0.0)
        kt = k * kk
        kt2 = kt * kt
        norm_kt2 = gl.sum(kt2.to(gl.float32), axis=0)
        norm_kt = gl.sqrt(norm_kt2 + 1e-12)
        okk = kt / norm_kt
        cdna3.buffer_store(okk.to(o_kk_ptr.dtype.element_ty), o_kk_ptr, offs.to(gl.int32), offs < T * C)
        ok = k * (1 + (a.to(gl.float32) - 1) * ka)
        okka = okk * a
        cdna3.buffer_store(ok.to(o_k_ptr.dtype.element_ty), o_k_ptr, offs.to(gl.int32), offs < T * C)
        cdna3.buffer_store(okka.to(o_kka_ptr.dtype.element_ty), o_kka_ptr, offs.to(gl.int32), offs < T * C)
