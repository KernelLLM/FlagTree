"""Final executable bytes, pure timed windows and exact native dependency roles."""
from dataclasses import asdict
from pathlib import Path
import re
import struct
import subprocess
from .model import QualificationResult, read_json, write_json
from .benchmarks import lookup, TARGETS
from .benchmarks.llvm import PHASES, window_counts

PAGE_FAILURE = 'timed instruction fetch page differs'


def require(condition, message):
    if not condition: raise ValueError(message)


def device_elf(data):
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    require(data.startswith(magic), 'not an offload bundle')
    count = struct.unpack_from('<Q', data, len(magic))[0]
    pos = len(magic) + 8
    for _ in range(count):
        offset, size, length = struct.unpack_from('<QQQ', data, pos)
        pos += 24
        name = data[pos:pos + length].decode()
        pos += length
        if name.startswith('maca-mxc-') and not name.endswith('-bc'):
            elf = data[offset:offset + size]
            require(elf.startswith(b'\x7fELF') and len(elf) == size, 'invalid device ELF')
            return elf
    raise ValueError('device ELF missing')


def functions(path):
    path = Path(path)
    data = path.read_bytes()
    sections = subprocess.check_output(['readelf', '-WS', str(path)], text=True)
    match = re.search(r'\]\s+\.text\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', sections)
    require(match, 'ELF text section missing')
    address, offset, size = (int(x, 16) for x in match.groups())
    symbols = subprocess.check_output(['readelf', '-Ws', str(path)], text=True)
    result = {}
    for addr, length, name in re.findall(r'^\s*\d+:\s+([0-9a-f]+)\s+(\d+)\s+FUNC\s+.*\s+(\S+)$', symbols, re.M):
        start = int(addr, 16) - address
        length = int(length)
        if 0 <= start and start + length <= size:
            result[name] = (int(addr, 16), data[offset + start:offset + start + length])
    return result


def assembly_records(text, symbol):
    body = text.split(symbol + ':', 1)[1].split('\t.section', 1)[0]
    result = []
    for line in body.splitlines():
        instruction = re.match(r'^\t([a-z][\w.]*)\s*(.*)', line)
        if instruction:
            result.append({
                'op': instruction[1], 'args': [x.strip() for x in instruction[2].split(',')] if instruction[2] else []
            })
    return result


def tuple_regs(base, width):
    require(re.fullmatch(r'r\d+', base), 'expected MTREG tuple: ' + base)
    return [f'r{int(base[1:])+i}' for i in range(width)]


def register_tuple(base, width):
    require(re.fullmatch(r'[rs]\d+', base), 'expected register tuple: ' + base)
    return [base[0] + str(int(base[1:]) + i) for i in range(width)]


def numeric_step(row, values, active):
    """Evaluate supported address/predicate instructions for the fixed 64-lane launch."""
    op, a = row['op'], row['args']

    def get(x):
        if x == 'xmsk': return tuple(active)
        if x in values: return values[x]
        try:
            return (int(x, 0) & 0xffffffff, ) * 64
        except ValueError:
            return (None, ) * 64

    if op in ('and_xmsk', 'or_xmsk'):
        old = tuple(active)
        rhs = get(a[1])
        require(None not in rhs, 'unknown execution predicate')
        active = [bool(x) and bool(y) if op == 'and_xmsk' else bool(x) or bool(y) for x, y in zip(old, rhs)]
        values[a[0]] = old
        return active
    result = None
    if op in ('mov_b32', 'smov_b32'): result = get(a[1])
    elif op in ('or3_b32', 'and_or_b32', 'and_xor_b32', 'shl_or_b32'):
        operands = [get(x) for x in a[1:]]
        result = tuple(
            None if None in (x, y,
                             z) else (x | y | z if op == 'or3_b32' else (x & y) | z if op == 'and_or_b32' else (x & y)
                                      ^ z if op == 'and_xor_b32' else (y << x) | z) & 0xffffffff
            for x, y, z in zip(*operands))
    elif op in ('bfe_i32', 'bfe_u32'):
        source, shift, width = (get(x) for x in a[1:])
        extracted = []
        for x, s, w in zip(source, shift, width):
            if None in (x, s, w) or s >= 32 or w > 32 - s:
                extracted.append(None)
                continue
            field = (x >> s) & ((1 << w) - 1)
            if op == 'bfe_i32' and w and field & (1 << (w - 1)): field -= 1 << w
            extracted.append(field & 0xffffffff)
        result = tuple(extracted)
    elif op in ('sor_b64', 'sxor_b64'):
        left, right = get(a[1]), get(a[2])
        result = tuple(None if x is None or y is None else bool(x) or bool(y) if op == 'sor_b64' else bool(x) ^ bool(y)
                       for x, y in zip(left, right))
    elif op in ('add_u32', 'add_i32', 'sub_u32', 'and_b32', 'or_b32', 'xor_b32', 'shl_b32', 'shr_b32', 'cmp_lt_u32',
                'cmp_eq_u32', 'cmp_gt_u32'):
        left, right = get(a[1]), get(a[2])
        out = []
        for x, y in zip(left, right):
            if x is None or y is None: v = None
            elif op in ('add_u32', 'add_i32'): v = x + y
            elif op == 'sub_u32': v = x - y
            elif op == 'and_b32': v = x & y
            elif op == 'or_b32': v = x | y
            elif op == 'xor_b32': v = x ^ y
            elif op == 'shl_b32': v = (y << x) if x < 32 else 0
            elif op == 'shr_b32': v = (y >> x) if x < 32 else 0
            elif op == 'cmp_lt_u32': v = x < y
            elif op == 'cmp_eq_u32': v = x == y
            else: v = x > y
            out.append(None if v is None else v & 0xffffffff)
        result = tuple(out)
    elif op == 'csel_b32':
        left, right, pred = get(a[1]), get(a[2]), get(a[3])
        result = tuple(None if p is None else x if p else y for x, y, p in zip(left, right, pred))
    elif op == 'shl_b64' and a[1:] in (['0', '0'], ['0x0', '0x0']):
        for reg in register_tuple(a[0], 2):
            values[reg] = (0, ) * 64
        return active
    if result is not None:
        if a[0] == 'xmsk':
            require(None not in result, 'unknown restored execution mask')
            return list(map(bool, result))
        values[a[0]] = result
    elif a and re.fullmatch(r'[rs]\d+', a[0]) and not op.startswith(('st', 'arrive', 'bra')):
        for reg in register_tuple(a[0], 2 if op == 'get_time' or '64' in op else 4 if '128' in op else 1):
            values.pop(reg, None)
    return active


def verify_point(spec, n, records):
    t = TARGETS[spec.target.name]
    kind = t['kind']
    w = t['width']
    independent = spec.metric == 'independent_issue'
    clocks = [i for i, r in enumerate(records) if r['op'] == 'get_time']
    require(len(clocks) == 2, 'exactly two counters required')
    start, stop = clocks
    timed = records[start + 1:stop]
    require(
        len(timed) == n and all(r['op'] == t['opcode'] for r in timed), 'timed window is not pure target opcode/count')
    require(
        not any(r['op'].startswith(('bra', 'br_', 'call', 'set_pc', 'and_xmsk', 'or_xmsk', 'set_xmsk'))
                for r in records), 'control flow or mask change')
    require(any(r['op'] == 'arrive' and r['args'] == ['gvmcnt(0) & bsmcnt(0)'] for r in records[:start]),
            'setup queues not drained')
    drain = next((i for i in range(stop + 1, len(records))
                  if records[i]['op'] == 'arrive' and records[i]['args'] == ['bsmcnt(0)']), None)
    require(
        drain is not None and any(r['op'] == 'arrive' and r['args'] == ['slcnt(0)'] for r in records[stop + 1:drain]),
        'post-counter recovery missing')
    numbers = {'r0': tuple(range(64))}
    active = [True] * 64
    for row in records[:start]:
        active = numeric_step(row, numbers, active)
    require(all(active), 'active mask changed')
    dests = []
    invariants = []
    chain = []
    for slot, row in enumerate(timed):
        a = row['args']
        dst = []
        inv = []
        if kind in ('lds', 'sts'):
            require(len(a) == 3 and a[2 if kind == 'lds' else 1] == '0x0', 'memory width/modifier changed')
            address = a[1 if kind == 'lds' else 0]
            if 'address_terms' in t:
                from .benchmarks.shared import lane_offsets
                want = tuple(slot * 4096 + x for x in lane_offsets(t['address_terms']))
            else:
                want = tuple((slot * 64 + lane) * t['size'] for lane in range(64))
            require(numbers.get(address) == want, 'shared address geometry/alignment not proven')
            dst = register_tuple(a[0], w) if kind == 'lds' else []
            inv = [address] + (register_tuple(a[2], w) if kind == 'sts' else [])
        elif kind == 'bpermute':
            require(len(a) == 4 and a[2] == '0x0', 'bpermute modifier changed')
            require(numbers.get(a[1]) == tuple((lane ^ 1) * 4 for lane in range(64)), 'bpermute byte address differs')
            dst = register_tuple(a[0], 1)
            inv = [a[1], a[3]]
        else:
            require(len(a) == (4 if kind in ('mma', 'perm') else 3), 'operand/modifier count differs')
            raw = a[3] if kind == 'mma' else a[0] if kind == 'fma' else a[1] if kind in (
                'integer', 'perm') and 'shl' not in t['opcode'] and 'shr' not in t['opcode'] else a[2]
            require(a[0] == raw, 'target RAW role differs')
            dst = tuple_regs(a[0], w)
            if kind == 'mma': inv = tuple_regs(a[1], 2) + tuple_regs(a[2], 2)
            elif kind == 'fma': inv = [a[1], a[2]]
            elif kind == 'perm':
                inv = [a[2], a[3]]
                require(numbers.get(a[3]) == (t['selector'], ) * 64, 'perm selector differs')
            elif kind == 'integer' and spec.conditions['immediate']:
                require(a[1] == '16', 'shift immediate differs')
            else:
                inv = [a[1] if kind in ('fadd', 'fmul') else a[2]]
            if not independent and chain: require(dst == chain[-1], 'dependent tuple chain broken')
            chain.append(dst)
        require(not set(dst) & set(inv), 'target overwrites invariant operand')
        if independent: require(not set(dst) & set(dests), 'independent outputs overlap')
        dests += dst
        invariants.append(inv)
    if independent:
        require(not set(dests) & {v for row in invariants for v in row}, 'independent result is a target input/address')
    if kind in ('fadd', 'fmul', 'fma', 'mma', 'integer', 'perm') and invariants:
        require(all(v == invariants[0] for v in invariants), 'invariant binding varies')
    return {
        'n': n, 'start_offset': start * 8, 'stop_offset': stop * 8, 'pc_offset':
        next(i * 8 for i, r in enumerate(records) if r['op'] == 'get_pc'), 'timed_opcode_counts':
        {t['opcode']: n} if n else {}, 'raw_tuples': chain, 'invariant_registers': invariants, 'timed_waits': [],
        'post_drain_offset': drain * 8
    }


def qualify(run):
    run = Path(run)
    saved = read_json(run / 'run.json')
    spec, family = lookup(saved['measurement']['measurement_id'])
    points = []
    try:
        require(saved['timing_protocol'] == 'short_straight_line', 'wrong timing protocol')
        counts = window_counts(spec)
        require({(c['n'], c['pad'])
                 for c in saved['cases']} == {(n, p)
                                              for n in counts
                                              for p in PHASES} and len(saved['cases']) == len(counts) * 4,
                'qualification count/layout coverage incomplete')
        for case in saved['cases']:
            out = run / case['path']
            (out / 'device.elf').write_bytes(device_elf((out / 'benchmark.fatbin').read_bytes()))
            address, code = functions(out / 'device.elf')['bench']
            obj = functions(out / 'reassembled.o')['bench'][1]
            require(code == obj[:len(code)], 'final ELF/object function bytes differ')
            asm = (out / 'benchmark.s').read_text()
            records = assembly_records(asm, 'bench')
            require(
                len(code) == 8 * len(records) and records[-1]['op'] == 'endk', 'function extent/native records differ')
            q = verify_point(spec, case['n'], records)
            require((address + q['start_offset']) // 4096 == (address + q['stop_offset'] + 7) // 4096, PAGE_FAILURE)
            q.update(
                pad=case['pad'], function_elf_address=address, function_bytes=len(code), resources={
                    key: re.search(r'\.' + key + r':\s*(\d+)', asm)[1]
                    for key in ('mtreg_count', 'streg_count', 'private_memory_size', 'share_memory_size',
                                'kernarg_size_bytes')
                }, instructions=[{'offset': i * 8, 'bytes': code[i * 8:i * 8 + 8].hex(), **r}
                                 for i, r in enumerate(records)])
            write_json(out / 'qualification.json', q)
            points.append(q)
        reference = functions(run / 'n0-p0/device.elf')['bench'][1]
        full = next(p for p in points if p['n'] == max(counts) and p['pad'] == 0)
        target = [r['bytes'] for r in full['instructions'] if full['start_offset'] < r['offset'] < full['stop_offset']]
        for p in points:
            code = functions(run / f'n{p["n"]}-p{p["pad"]}/device.elf')['bench'][1]
            timed = [r for r in p['instructions'] if p['start_offset'] < r['offset'] < p['stop_offset']]
            pads = set(range(p['pc_offset'] + 8, p['pc_offset'] + 8 + p['pad'] * 8, 8))
            require(
                all(p['instructions'][i // 8]['op'] == 'snop' and p['instructions'][i // 8]['args'] == ['0']
                    for i in pads), 'padding position differs')
            remove = {r['offset'] for r in timed} | pads
            require(b''.join(code[i:i + 8] for i in range(0, len(code), 8) if i not in remove) == reference,
                    'non-target code varies with N/layout')
            require(p['resources'] == points[0]['resources'], 'resources/ABI vary with N/layout')
            require([r['bytes'] for r in timed] == target[:p['n']], 'target operand binding varies with N/layout')
        result = QualificationResult(
            True, None, {
                'native_opcode': spec.target.expected_isa, 'pure_target_window': True, 'matched_function_bytes': True,
                'points': len(points), 'timing_protocol': 'short_straight_line'
            })
    except (ValueError, KeyError, IndexError) as error:
        result = QualificationResult(False, str(error), {'points': len(points)})
    write_json(run / 'qualification.json', asdict(result))
    return result


def verify_qualification(run):
    run = Path(run)
    require(read_json(run / 'qualification.json')['passed'], 'qualification failed')
    return read_json(run / 'run.json')
