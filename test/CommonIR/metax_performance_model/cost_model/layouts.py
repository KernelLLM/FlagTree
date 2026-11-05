"""Single-CTA GEMM shapes, explicit TTGIR encodings and host reference inputs."""
import ctypes as C
import struct
import re

WORDS = 16384

COMPILE_OPTIONS = dict(num_warps=1, num_stages=1, enable_fp_fusion=False, pipeline='basic')
TARGET = dict(backend='maca', arch=80, warp_size=64)
SAMPLING = dict(processes=2, warmup=8, samples=64, shuffle_seed=1700, repeatability_percent=5, timeout_seconds=300)
# Only these measurements are eligible for end-to-end publication; each still
# requires fresh qualification and timing.
FORMAL_MEASUREMENTS = (
    'byte_perm.back.dependent_latency',
    'byte_perm.back.independent_issue',
    'byte_perm.front.dependent_latency',
    'byte_perm.front.independent_issue',
    'fadd_f32.dependent_latency',
    'fadd_f32.independent_issue',
    'fma_f32.dependent_latency',
    'fma_f32.independent_issue',
    'fmul_f32.dependent_latency',
    'fmul_f32.independent_issue',
    'i32_add.varying.dependent_latency',
    'i32_add.varying.independent_issue',
    'i32_or.dependent_latency',
    'i32_xor.dependent_latency',
    'i32_xor.independent_issue',
    'lds_b128.independent_issue',
    'lds_b64.independent_issue',
    'lds_gemm.map10.independent_issue',
    'lds_gemm.map2.independent_issue',
    'lds_gemm.map3.independent_issue',
    'lds_gemm.map5.independent_issue',
    'lds_gemm.map6.independent_issue',
    'lds_gemm.map7.independent_issue',
    'lds_gemm.map8.independent_issue',
    'lds_gemm.map9.independent_issue',
    'lds_u16.independent_issue',
    'lshr_i32.imm16.dependent_latency',
    'lshr_i32.imm16.independent_issue',
    'mma_f16_16x16x16.dependent_latency',
    'mma_f16_16x16x16.independent_issue',
    'shl_i32.imm16.dependent_latency',
    'shl_i32.imm16.independent_issue',
    'sts_b128.independent_issue',
    'sts_b16.independent_issue',
    'sts_b32.independent_issue',
    'sts_b64.independent_issue',
)

SHAPES = ((16, 32, 32), (32, 32, 64), (32, 64, 64), (64, 32, 128))
LAYOUTS = (
    dict(name='plain', size_per_thread=[1, 1], threads_per_warp=[2, 32], shared=[1, 1, 1], elements_mnk=[1, 1, 4],
         col_major=0),
    dict(name='swizzle', size_per_thread=[1, 1], threads_per_warp=[2, 32], shared=[4, 2, 4], elements_mnk=[1, 1, 4],
         col_major=0),
    dict(name='blocked', size_per_thread=[1, 2], threads_per_warp=[4, 16], shared=[4, 2, 4], elements_mnk=[1, 1, 4],
         col_major=0),
    dict(name='k8', size_per_thread=[1, 1], threads_per_warp=[2, 32], shared=[4, 2, 4], elements_mnk=[1, 1, 8],
         col_major=0),
    dict(name='column', size_per_thread=[1, 1], threads_per_warp=[2, 32], shared=[4, 2, 4], elements_mnk=[1, 1, 4],
         col_major=1),
    dict(name='b-kcontig', size_per_thread=[1, 1], threads_per_warp=[2, 32], shared=[4, 2, 4], elements_mnk=[1, 1, 4],
         col_major=0, b_order=[0, 1]),
)


def candidates():
    return [
        dict(id=f'm{m}-n{n}-k{k}-{layout["name"]}', shape=[m, n, k], block_k=32, a_layout='row', b_layout='row',
             layout=layout) for m, n, k in SHAPES for layout in LAYOUTS
    ]


def buffer_layout(variant):
    m, n, k = variant['shape']
    b = m * k // 2
    c = b + k * n // 2
    return dict(b=b, c=c, ticks=m * n, start=m * n + 128, stop=m * n + 256)


def apply_layout(text, variant):
    """Replace encoding definitions, leaving operations and global addresses intact."""
    cfg = variant['layout']
    b_shared = re.findall(r'!ttg.memdesc<[^\n>]*, (#\w+), #\w+> ->[^\n]*opIdx = 1', text)
    if 'b_order' in cfg and not b_shared:
        raise ValueError('B DotOperand shared encoding is unavailable')

    def replace(match):
        name, kind, body = match.groups()
        if kind == 'blocked' and 'sizePerThread = [' in body and 'threadsPerWarp = [64]' not in body:
            body = f'sizePerThread = {cfg["size_per_thread"]}, threadsPerWarp = {cfg["threads_per_warp"]}, warpsPerCTA = [1, 1], order = [1, 0]'
        elif kind == 'swizzled_shared':
            vec, per, maximum = cfg['shared']
            order = cfg.get('b_order', [1, 0]) if name in b_shared else [1, 0]
            body = f'vec = {vec}, perPhase = {per}, maxPhase = {maximum}, order = {order}'
        elif kind == 'maca_mma':
            body = re.sub(r'elementsMNK = \[[^]]+\]', f'elementsMNK = {cfg["elements_mnk"]}', body)
            body = re.sub(r'colMajor = \d+', f'colMajor = {cfg["col_major"]}', body)
        return f'{name} = #ttg.{kind}<{{{body}}}>'

    return re.sub(r'^(#\w+) = #ttg\.(blocked|swizzled_shared|maca_mma)<\{(.*)\}>$', replace, text, flags=re.M)


def inputs_for(variant, pattern):
    m, n, kdim = variant['shape']
    offsets = buffer_layout(variant)
    values = (C.c_uint32 * WORDS)()
    a = [[1.0 if pattern == 0 else float((i + 2 * k) % 5 - 2) for k in range(kdim)] for i in range(m)]
    b = [[1.0 if pattern == 0 else float((2 * k + j) % 3 - 1) for j in range(n)] for k in range(kdim)]
    c = [[1.0 if pattern == 0 else float((i + j) % 5 - 2) for j in range(n)] for i in range(m)]
    for matrix, offset, layout in ((a, 0, variant['a_layout']), (b, offsets['b'], variant['b_layout'])):
        rows, cols = len(matrix), len(matrix[0])
        ordered = [matrix[i][j] for i in range(rows) for j in range(cols)
                   ] if layout == 'row' else [matrix[i][j] for j in range(cols) for i in range(rows)]
        raw = b''.join(struct.pack('<e', v) for v in ordered)
        for i in range(len(raw) // 4):
            values[offset + i] = struct.unpack_from('<I', raw, i * 4)[0]
    for i in range(m):
        for j in range(n):
            values[offsets['c'] + i * n + j] = struct.unpack('<I', struct.pack('<f', c[i][j]))[0]
    return values, a, b, c
