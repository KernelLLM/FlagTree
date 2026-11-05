"""Fixed straight-line native instruction windows and exact host oracles."""
import struct

WORDS = 8192
COUNTS = (0, 1, 2, 3, 4, 5, 6, 8)
PHASES = (0, 1, 2, 3)


def window_counts(spec):
    return tuple(n for n in COUNTS if spec.metric != 'independent_issue' or n <= spec.independent_chains)


def float_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def bits_float(x):
    return struct.unpack('<f', struct.pack('<I', x))[0]


def render(spec, n, pad):
    from . import TARGETS
    t = TARGETS[spec.target.name]
    kind = t['kind']
    w = t['width']
    k = spec.independent_chains
    floating = kind in ('fadd', 'fmul', 'fma', 'mma')
    element = 'float' if floating else 'i32'
    typ = element if w == 1 else f'<{w} x {element}>'
    lines = []

    def emit(expr):
        v = f'%v{len(lines)}'
        lines.append(f'  {v} = {expr}')
        return v

    def void(expr):
        lines.append('  ' + expr)

    def gep(base, index):
        return emit(f'getelementptr i32,ptr addrspace(1) {base},i32 {index}')

    def pack(parts):
        if w == 1: return parts[0]
        value = 'undef'
        for j, p in enumerate(parts):
            value = emit(f'insertelement {typ} {value},{element} {p},i32 {j}')
        return value

    def unpack(value):
        return [value] if w == 1 else [emit(f'extractelement {typ} {value},i32 {j}') for j in range(w)]

    def shared_ptr(addr):
        return emit(f'inttoptr i32 {addr} to ptr addrspace(3)')

    def shared_store(value, addr):
        styp = typ
        if t['size'] == 2:
            styp = 'i16'
            value = emit(f'trunc i32 {value} to i16')
        void(f'store volatile {styp} {value},ptr addrspace(3) {shared_ptr(addr)},align {t["size"]}')

    lane = emit('call i32 @llvm.mxc.thread.id.x()')
    base = emit(f'shl i32 {lane},6')
    lane_address = None
    if 'address_terms' in t:
        lane_address = '0'
        for mask, coefficient in t['address_terms']:
            masked = emit(f'and i32 {lane},{mask}')
            pred = emit(f'icmp eq i32 {masked},{mask}')
            contribution = emit(f'select i1 {pred},i32 {coefficient},i32 0')
            lane_address = emit(f'add i32 {lane_address},{contribution}')
    initial = []
    data = []
    addresses = []
    scratch = emit('ptrtoint ptr addrspace(3) @scratch to i32') if kind in ('lds', 'sts') else None
    for c in range(k):
        parts = []
        old = []
        for j in range(w):
            if lane_address is None: ix = emit(f'add i32 {base},{c*w+j}')
            else:
                word = emit(f'lshr i32 {lane_address},2')
                ix = emit(f'add i32 {word},{c*1024+j}')
            v = emit(f'load volatile i32,ptr addrspace(1) {gep("%in",ix)},align 4')
            if lane_address is not None and t['size'] == 2:
                half = emit(f'and i32 {lane_address},2')
                shift = emit(f'shl i32 {half},3')
                v = emit(f'lshr i32 {v},{shift}')
            parts.append(emit(f'bitcast i32 {v} to float') if floating else v)
            if scratch or kind == 'bpermute':
                ix = emit(f'add i32 {base},{32+c*w+j}')
                old.append(emit(f'load volatile i32,ptr addrspace(1) {gep("%in",ix)},align 4'))
        data.append(pack(parts))
        initial.append(pack(old) if kind in ('lds', 'bpermute') else data[-1])
        if scratch:
            addr = emit(f'mul i32 {lane},{t["size"]}') if lane_address is None else lane_address
            addr = emit(f'add i32 {addr},{c*64*t["size"] if lane_address is None else c*4096}')
            addr = emit(f'add i32 {scratch},{addr}')
            addresses.append(addr)
            shared_store(data[-1] if kind == 'lds' else pack(old), addr)
    extra = []
    constraints = ['=&r'] * k + [str(i) for i in range(k)]

    def argument(typ, value, constraint='r'):
        index = 2 * k + len(extra)
        extra.append(f'{typ} {value}')
        constraints.append(constraint)
        return index

    def load_arg(typ, index, align):
        return emit(f'load volatile {typ},ptr addrspace(1) {gep("%in",index)},align {align}')

    if kind == 'mma':
        a = argument('<4 x half>', load_arg('<4 x half>', 4096, 8))
        b = argument('<4 x half>', load_arg('<4 x half>', 4100, 8))
    elif kind in ('fadd', 'fmul', 'fma'):
        a = argument('float', load_arg('float', 4096, 4))
        if kind == 'fma': b = argument('float', load_arg('float', 4100, 4))
    elif kind == 'integer' and not spec.conditions['immediate']: a = argument('i32', load_arg('i32', 4096, 4))
    elif kind == 'perm':
        a = argument('i32', load_arg('i32', 4096, 4))
        sel = emit(f'call i32 asm sideeffect "smov_b32 $0, {t["selector"]:#x}", "=s,~{{memory}}"()')
        selector = argument('i32', sel, 's')
    elif scratch:
        addrids = [argument('i32', addr) for addr in addresses]
    elif kind == 'bpermute':
        peer = emit(f'xor i32 {lane},1')
        addr = emit(f'shl i32 {peer},2')
        addrid = argument('i32', addr)
        sources = [argument('i32', v) for v in data]
    void('call void asm sideeffect "arrive gvmcnt(0) & bsmcnt(0)", "~{memory}"()')
    pc = emit('call i64 asm sideeffect "get_pc $0", "=s,~{memory}"()')
    if pad: void('call void asm sideeffect "' + '\\0A'.join(['snop 0'] * pad) + '", "~{memory}"()')
    start = emit('call i64 @llvm.readcyclecounter()')
    ops = []
    for i in range(n):
        c = i if spec.metric == 'independent_issue' else 0
        op = t['opcode']
        if kind == 'mma': s = f'{op} ${c}, ${a}, ${b}, ${c}'
        elif kind == 'fma': s = f'{op} ${c}, ${a}, ${b}'
        elif kind in ('fadd', 'fmul'): s = f'{op} ${c}, ${a}, ${c}'
        elif kind == 'integer':
            rhs = '16' if spec.conditions['immediate'] else f'${a}'
            s = f'{op} ${c}, {rhs}, ${c}' if 'shr' in op or 'shl' in op else f'{op} ${c}, ${c}, {rhs}'
        elif kind == 'perm':
            s = f'{op} ${c}, ${c}, ${a}, ${selector}'
        elif kind == 'lds':
            s = f'{op} ${c}, ${addrids[c]}, 0x0'
        elif kind == 'sts':
            s = f'{op} ${addrids[c]}, 0x0, ${c}'
        else:
            s = f'{op} ${c}, ${addrid}, 0x0, ${sources[c]}'
        ops.append(s)
    result_type = typ if k == 1 else '{ ' + ', '.join([typ] * k) + ' }'
    body = '\\0A'.join(ops)
    bundle = emit(f'call {result_type} asm sideeffect "{body}", "' + ','.join(constraints) + ',~{memory}"(' +
                  ','.join([f'{typ} {v}' for v in initial] + extra) + ')')
    stop = emit('call i64 @llvm.readcyclecounter()')
    void(f'call void asm sideeffect "arrive bsmcnt(0)", "s,~{{memory}}"(i64 {stop})')
    results = [bundle] if k == 1 else [emit(f'extractvalue {result_type} {bundle},{c}') for c in range(k)]

    def output(values):
        for c, value in enumerate(values):
            for j, v in enumerate(unpack(value)):
                if floating: v = emit(f'bitcast float {v} to i32')
                ix = emit(f'add i32 {base},{16+c*w+j}')
                void(f'store volatile i32 {v},ptr addrspace(1) {gep("%out",ix)},align 4')

    output(results)
    if kind == 'sts':
        readback = []
        for addr in addresses:
            styp = 'i16' if t['size'] == 2 else typ
            v = emit(f'load volatile {styp},ptr addrspace(3) {shared_ptr(addr)},align {t["size"]}')
            readback.append(emit(f'zext i16 {v} to i32') if t['size'] == 2 else v)
        output(readback)
    delta = emit(f'sub i64 {stop},{start}')
    for index, v in ((0, delta), (2, start), (4, stop), (6, pc)):
        ix = emit(f'add i32 {base},{index}')
        void(f'store volatile i64 {v},ptr addrspace(1) {gep("%out",ix)},align 8')
    void('ret void')
    header = 'target triple = "mxc-metax-macahca"\ndeclare i64 @llvm.readcyclecounter()\ndeclare i32 @llvm.mxc.thread.id.x()\n'
    header += 'attributes #0 = { "target-cpu"="xcore1000" "metaxgpu-flat-work-group-size"="64,64" "metaxgpu-implicitarg-num-bytes"="56" "uniform-work-group-size"="true" }\n'
    if scratch:
        header += f'@scratch = internal addrspace(3) global [{2048 if lane_address is None else 8192} x i32] undef,align 16\n'
    return header + 'define protected metaxgpu_kernel void @bench(ptr addrspace(1) %in,ptr addrspace(1) %out) #0 align 4096 {\nentry:\n' + '\n'.join(
        lines) + '\n}\n'


def seed(lane, c, j, pattern, floating):
    return float_bits(1 + lane / 4 + c + j +
                      pattern * 2) if floating else (0x12345678 ^ lane * 0x10101 ^ c * 0x110011 ^ j * 0x11111111
                                                     ^ pattern * 0xa5a5a5a5) & 0xffffffff


def expected(spec, n, inp, lane, c, j):
    from . import TARGETS
    t = TARGETS[spec.target.name]
    kind = t['kind']
    w = t['width']
    index = lane * 64 + c * w + j
    steps = (int(c < n) if spec.metric == 'independent_issue' else n)
    bits = int(inp[index])
    value = bits_float(bits) if kind in ('fadd', 'fmul', 'fma', 'mma') else bits
    if kind == 'lds':
        if 'address_terms' in t and steps:
            from .shared import lane_offsets
            offset = lane_offsets(t['address_terms'])[lane] + c * 4096
            return (int(inp[offset // 4 + j]) >> ((offset % 4) * 8)) & (0xffff if t['size'] == 2 else 0xffffffff)
        value = bits if steps else int(inp[lane * 64 + 32 + c * w + j])
        return value & (0xffff if t['size'] == 2 and steps else 0xffffffff)
    if kind == 'sts':
        return (bits if steps else int(inp[lane * 64 + 32 + c * w + j])) & (0xffff if t['size'] == 2 else 0xffffffff)
    if kind == 'bpermute': return int(inp[(lane ^ 1) * 64 + c]) if steps else int(inp[lane * 64 + 32 + c])
    for _ in range(steps):
        if kind in ('fadd', 'fmul', 'fma', 'mma'):
            rhs = bits_float(int(inp[4096])) if kind != 'mma' else 16.
            value = value * rhs if kind == 'fmul' else value + rhs * bits_float(int(
                inp[4100])) if kind == 'fma' else value + rhs
            value = bits_float(float_bits(value))
        elif kind == 'perm':
            packed = int(inp[4096]) | (value << 32)
            value = sum(((packed >> ((t['selector'] >> (8 * i) & 7) * 8)) & 255) << (8 * i) for i in range(4))
        else:
            rhs = t['rhs']
            op = t['opcode']
            value = (value + rhs if op == 'add_u32' else value & rhs if op == 'and_b32' else value
                     | rhs if op == 'or_b32' else value
                     ^ rhs if op == 'xor_b32' else value << rhs if op == 'shl_b32' else value >> rhs) & 0xffffffff
    return float_bits(value) if kind in ('fadd', 'fmul', 'fma', 'mma') else value
