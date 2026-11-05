import triton
import triton.language as tl


@triton.jit
def layout_gemm(Input, Output, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    prime = tl.inline_asm_elementwise("get_time $0\narrive slcnt(0)", "=s,~{memory}", [], dtype=tl.uint64,
                                      is_pure=False, pack=1)
    start = tl.inline_asm_elementwise("get_time $0\narrive slcnt(0)", "=s,s,~{memory}", [prime], dtype=tl.uint64,
                                      is_pure=False, pack=1)
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    ks = tl.arange(0, BLOCK_K)
    a_ptr = Input.to(tl.pointer_type(tl.float16))
    b_ptr = (Input + M * K // 2).to(tl.pointer_type(tl.float16))
    c_ptr = (Input + (M * K + K * N) // 2).to(tl.pointer_type(tl.float32))
    acc = tl.load(c_ptr + rows[:, None] * N + cols[None, :])
    for k0 in tl.static_range(0, K, BLOCK_K):
        a_offsets = rows[:, None] * K + k0 + ks[None, :]
        b_offsets = (k0 + ks[:, None]) * N + cols[None, :]
        a = tl.load(a_ptr + a_offsets)
        b = tl.load(b_ptr + b_offsets)
        acc = tl.dot(a, b, acc)
    tl.store(Output.to(tl.pointer_type(tl.float32)) + rows[:, None] * N + cols[None, :], acc)
    stop = tl.inline_asm_elementwise("arrive gvmcnt(0)\nget_time $0\narrive slcnt(0)", "=s,s,~{memory}", [start],
                                     dtype=tl.uint64, is_pure=False, pack=1)
    lane = tl.arange(0, 64)
    tl.store((Output + M * N).to(tl.pointer_type(tl.uint64)) + lane, stop - start)
    tl.store((Output + M * N + 128).to(tl.pointer_type(tl.uint64)) + lane, start)
    tl.store((Output + M * N + 256).to(tl.pointer_type(tl.uint64)) + lane, stop)
