"""LLVM operations, lane addresses and grouped issue/dependency events."""
from collections import Counter
import re
import math
import struct

from measurement.qualify import require

SSA = r'%[\w.$-]+'
MMA = 'llvm.mxc.mma.f32.16x16x16f16'
TYPE = r'<\d+ x (?:i\d+|half|float)>|i\d+|half|float'


def llvm_nodes(text, name='layout_gemm'):
    match = re.search(r'^define [^\n]*@' + re.escape(name) + r'\([^\n]*\)[^\n]*\{\n.*?^}', text, re.M | re.S)
    require(match is not None, 'LLVM kernel is missing: ' + name)
    nodes = []
    definitions = {}
    for line in match[0].splitlines()[1:-1]:
        line = line.strip()
        if not line or line.endswith(':') or line.startswith(';'): continue
        assignment = re.match('(' + SSA + r') = (.*)', line)
        result, operation = assignment.groups() if assignment else (None, line)
        uses = re.findall(SSA, operation)
        nodes.append({
            'id': len(nodes), 'result': result, 'llvm': operation, 'opcode': operation.split()[0], 'inputs': uses,
            'predecessors': sorted({definitions[u]
                                    for u in uses
                                    if u in definitions})
        })
        if result: definitions[result] = nodes[-1]['id']
    clocks = [n['id'] for n in nodes if 'asm sideeffect "' in n['llvm'] and 'get_time' in n['llvm']]
    require(len(clocks) == 3, 'expected prime/start/stop in LLVM')
    require(not any(n['opcode'] in ('br', 'phi', 'switch', 'invoke') for n in nodes),
            'timeline requires an unrolled single basic block')
    return nodes, clocks[1], clocks[2]


def memory_form(operation):
    intrinsic = re.search(r'@(llvm\.mxc\.(?:ldg|stg)\.predicator(?:\.bsm)?\.\w+)\(', operation)
    if intrinsic:
        offset = re.search(r'\(ptr addrspace\(1\) [^,]+, i32 (-?\d+),', operation)
        return {
            'kind':
            'async_copy' if '.bsm.' in intrinsic[1] else 'global_load' if '.ldg.' in intrinsic[1] else 'global_store',
            'form': intrinsic[1], 'address_space': 1, 'flags': re.findall(r'i1 (true|false)', operation),
            'immediate_offset': int(offset[1]) if offset else None
        }
    memory = re.match(r'(load|store) (volatile )?(<\d+ x (?:i\d+|half|float)>|i\d+|half|float)', operation)
    if not memory: return None
    address = re.search(r'ptr addrspace\((\d+)\)', operation)
    space = int(address[1]) if address else 0
    alignment = re.search(r'align (\d+)', operation)
    return {
        'kind': ('shared_' if space == 3 else 'global_' if space == 1 else 'other_') +
        ('load' if memory[1] == 'load' else 'store'), 'form':
        memory[3], 'address_space':
        space, 'volatile':
        bool(memory[2]), 'alignment':
        int(alignment[1]) if alignment else None
    }


def type_bytes(typ):
    vector = re.fullmatch(r'<(\d+) x (.+)>', typ)
    if vector:
        return int(vector[1]) * type_bytes(vector[2])
    return {'half': 2, 'float': 4, 'double': 8}.get(typ) or int(typ[1:]) // 8


def address_values(nodes):
    """Evaluate integer SSA and byte offsets; pointer bases stay separate."""
    values = {}
    pointers = {'%0': ('input', [0] * 64), '%1': ('output', [0] * 64), '@global_smem': ('shared', [0] * 64)}

    def value(token):
        token = token.strip()
        if token in values:
            return values[token]
        try:
            return [int(token, 0)] * 64
        except ValueError:
            return None

    for node in nodes:
        op, result = node['llvm'], node['result']
        if result is None:
            continue
        out = None
        if op == 'call i32 @llvm.mxc.thread.id.x()':
            out = list(range(64))
        binary = re.fullmatch(r'(add|sub|mul|and|or|xor|shl|lshr|ashr)(?: nuw| nsw| exact)* i(\d+) ([^,]+), (.+)', op)
        if binary:
            inst, width, a, b = binary.groups()
            left, right = value(a), value(b)
            if left is not None and right is not None:
                bits = int(width)
                mask = (1 << bits) - 1
                out = []
                for x, y in zip(left, right):
                    if inst == 'add': z = x + y
                    elif inst == 'sub': z = x - y
                    elif inst == 'mul': z = x * y
                    elif inst == 'and': z = x & y
                    elif inst == 'or': z = x | y
                    elif inst == 'xor': z = x ^ y
                    elif inst == 'shl': z = x << y
                    elif inst == 'lshr': z = (x & mask) >> y
                    else: z = ((x & mask) - (1 << bits) if x & (1 << (bits - 1)) else x) >> y
                    out.append(z & mask)
        comparison = re.fullmatch(r'icmp (eq|ne|ult|ule|ugt|uge) i\d+ ([^,]+), (.+)', op)
        if comparison:
            relation, a, b = comparison.groups()
            left, right = value(a), value(b)
            if left is not None and right is not None:
                out = [
                    dict(eq=x == y, ne=x != y, ult=x < y, ule=x <= y, ugt=x > y, uge=x >= y)[relation]
                    for x, y in zip(left, right)
                ]
        selection = re.fullmatch(r'select i1 ([^,]+), i\d+ ([^,]+), i\d+ (.+)', op)
        if selection:
            pred, a, b = map(value, selection.groups())
            if pred is not None and a is not None and b is not None:
                out = [x if p else y for p, x, y in zip(pred, a, b)]
        cast = re.fullmatch(r'(zext|trunc|sext) i(\d+) ([^ ]+) to i(\d+)', op)
        if cast:
            kind, old, token, new = cast.groups()
            a = value(token)
            if a is not None:
                out = [((x - (1 << int(old)) if kind == 'sext' and x & (1 << (int(old) - 1)) else x)
                        & ((1 << int(new)) - 1)) for x in a]
        gep = re.fullmatch(r'getelementptr(?: inbounds)? (i\d+|half|float), ptr addrspace\(\d+\) (.+), i\d+ ([^,]+)',
                           op)
        if gep:
            typ, base, index = gep.groups()
            # The compiler uses one constant inner GEP for shared allocations.
            inner = re.fullmatch(r'getelementptr \(i8, ptr addrspace\(3\) @global_smem, i32 (\d+)\)', base)
            source = ('shared', [int(inner[1])] * 64) if inner else pointers.get(base)
            offsets = value(index)
            if source is not None and offsets is not None:
                pointers[result] = (source[0], [x + type_bytes(typ) * y for x, y in zip(source[1], offsets)])
        if out is not None:
            values[result] = out
    return values, pointers


def memory_accesses(nodes, begin, end, pointers):
    accesses = []
    for node in nodes[begin + 1:end]:
        form = memory_form(node['llvm'])
        if form is None:
            continue
        pointer = re.search(r'ptr addrspace\(\d+\) (%[\w.$-]+)', node['llvm'])
        source = pointers.get(pointer[1]) if pointer else None
        if not form['form'].startswith('llvm.'):
            width = type_bytes(form['form'])
        else:
            typ = re.match(r'call (' + TYPE + ') ', node['llvm']) if form['kind'] == 'global_load' else re.search(
                r'\(ptr addrspace\(1\) [^,]+, i32 [^,]+, (' + TYPE + ') ', node['llvm'])
            require(typ is not None, 'unsupported global intrinsic width: ' + node['llvm'])
            width = type_bytes(typ[1])
        offsets = source[1] if source else None
        accesses.append({
            'node': node['id'], 'result': node['result'], **form, 'access_bytes': width, 'allocation':
            source[0] if source else None, 'byte_offsets': offsets, 'relative_lane_offsets':
            [x - offsets[0] for x in offsets] if offsets is not None else None, 'unique_lane_addresses':
            len(set(offsets)) if offsets is not None else None, 'unresolved':
            [] if source else ['lane addresses not evaluated']
        })
    return accesses


def footprint(access):
    offsets = access['byte_offsets']
    require(offsets is not None, 'shared alias footprint is unavailable')
    return {i for base in offsets for i in range(base, base + access['access_bytes'])}


def source_locations(text):
    locations = dict(re.findall(r'^(#loc\d*) = loc\((.*)\)$', text, re.M))

    def resolve(token):
        text = locations[token]
        source = re.search(r'"([^"]+)":(\d+):(\d+)', text)
        if source: return source[0]
        parents = re.findall(r'#loc\d*', text)
        return resolve(parents[0]) if parents else 'unknown'

    return {token: resolve(token) for token in locations}


def source_tags(mlir):
    locations = source_locations(mlir)
    tags = []
    active = False
    for line in mlir.splitlines():
        if '%start = llvm.inline_asm' in line:
            active = True
            continue
        if '%stop = llvm.inline_asm' in line: break
        if not active: continue
        kind = None
        if 'llvm.call @llvm.mxc.ldg.predicator.' in line: kind = 'global_load'
        elif 'llvm.call @llvm.mxc.stg.predicator.' in line: kind = 'global_store'
        elif 'llvm.call @llvm.mxc.mma.' in line: kind = 'mma'
        elif 'llvm.load ' in line and '!llvm.ptr<3>' in line: kind = 'shared_load'
        elif 'llvm.store ' in line and '!llvm.ptr<3>' in line: kind = 'shared_store'
        if kind:
            location = re.search(r'loc\((#loc\d*)\)\s*$', line)
            require(location is not None, 'selected MLIR source location missing')
            tags.append((kind, locations[location[1]]))
    return iter(tags)


def operand_evidence(nodes, pointers, integer_values, input_words=None, active_lanes=None):
    """Propagate known input bytes and constants; unknown bytes remain unknown."""
    values = {}
    shared = {}
    evidence = {}
    source = b''.join(struct.pack('<I', word) for word in input_words) if input_words is not None else None

    def literal(typ, token):
        size = type_bytes(typ)
        if token in values:
            return values[token]
        if token in integer_values and typ.startswith('i'):
            return [tuple((x % (1 << (size * 8))).to_bytes(size, 'little')) for x in integer_values[token]]
        vector = re.fullmatch(r'<(\d+) x (.+)>', typ)
        if token == 'zeroinitializer':
            raw = (0, ) * size
        elif vector and token.startswith('<'):
            parts = arguments(token[1:-1])
            if len(parts) != int(vector[1]):
                return [(None, ) * size] * 64
            return [sum((literal(t, v)[lane] for t, v in parts), ()) for lane in range(64)]
        elif typ.startswith('i') and re.fullmatch(r'-?\d+', token):
            raw = tuple((int(token) % (1 << (size * 8))).to_bytes(size, 'little'))
        elif typ in ('half', 'float') and token not in ('undef', 'poison'):
            try:
                number = (struct.unpack('>e', bytes.fromhex(token[3:]))[0] if token.startswith('0xH') else
                          struct.unpack('>d', bytes.fromhex(token[2:]))[0] if token.startswith('0x') else float(token))
                raw = tuple(struct.pack('<e' if typ == 'half' else '<f', number))
            except (ValueError, OverflowError, struct.error):
                raw = (None, ) * size
        else:
            raw = (None, ) * size
        return [raw] * 64

    def arguments(text):
        token = r'%[\w.$-]+|zeroinitializer|undef|poison|<[^<>]*>|0x[0-9A-Fa-fH]+|-?[0-9.eE+]+'
        return re.findall('(' + TYPE + r') (' + token + ')', text)

    def numbers(typ, raw):
        vector = re.fullmatch(r'<(\d+) x (.+)>', typ)
        element = vector[2] if vector else typ
        size = type_bytes(element)
        result = []
        for lane in raw:
            for i in range(0, len(lane), size):
                part = lane[i:i + size]
                if None in part:
                    return None
                result.append(
                    struct.unpack('<e' if element == 'half' else '<f', bytes(part))[0] if element in (
                        'half', 'float') else int.from_bytes(bytes(part), 'little'))
        return result

    for node in nodes:
        op, result = node['llvm'], node['result']
        args = arguments(op)
        binary_float = re.fullmatch(r'(fadd|fmul) float ([^,]+), (.+)', op)
        if binary_float:
            args = [('float', binary_float[2]), ('float', binary_float[3])]
        operands = [literal(typ, value) for typ, value in args]
        access = memory_form(op)
        raw = None
        if access is not None:
            pointer = re.search(r'ptr addrspace\(\d+\) (' + SSA + r')', op)
            location = pointers.get(pointer[1]) if pointer else None
            if access['kind'] == 'global_load' and access['form'].startswith('llvm.'):
                mask = re.search(r', i64 (' + SSA + r'|-?\d+)', op)
                mask_def = next((n['llvm'] for n in nodes if n['result'] == mask[1]), '') if mask else ''
                # LoadStoreOpToLLVM emits this full-wave predicate for an unmasked load.
                full_mask = mask is not None and (mask[1] in ('-1', str((1 << 64) - 1)) or mask_def
                                                  == 'call i64 @llvm.mxc.icmp.i64.i32(i32 1, i32 0, i32 34)')
                if (not full_mask or access['immediate_offset'] != 0 or access['flags']
                        not in (['true', 'true', 'false', 'false'], ['true', 'false', 'false', 'false'])):
                    location = None
            typ = args[0][0] if access['kind'].endswith('store') else re.match(
                r'(?:load (?:volatile )?|call )(' + TYPE + ')', op)[1]
            size = type_bytes(typ)
            if location is not None:
                allocation, offsets = location
                if access['kind'].endswith('load'):
                    raw = []
                    for offset in offsets:
                        raw.append(
                            tuple(source[offset:offset + size]) if allocation == 'input' and source is not None
                            and 0 <= offset <= len(source) - size else tuple(
                                shared.get(i)
                                for i in range(offset, offset + size)) if allocation == 'shared' else (None, ) * size)
                elif allocation == 'shared':
                    writes = {}
                    for offset, data in zip(offsets, literal(typ, args[0][1])):
                        for i, byte in enumerate(data, offset):
                            writes.setdefault(i, set()).add(byte)
                    for offset, candidates in writes.items():
                        shared[offset] = next(iter(candidates)) if len(candidates) == 1 else None
                elif allocation == 'input' and access['kind'].endswith('store'):
                    source = None
            elif access['kind'].endswith('store'):
                if access['address_space'] == 3:
                    shared.clear()
                else:
                    source = None
        elif node['opcode'] in ('bitcast', 'freeze') and operands:
            raw = operands[0]
        elif node['opcode'] == 'insertelement' and len(args) == 3:
            width = type_bytes(args[0][0])
            size = type_bytes(args[1][0])
            index = int(args[2][1])
            raw = [
                tuple(a[:index * size]) + tuple(b) + tuple(a[(index + 1) * size:width])
                for a, b in zip(operands[0], operands[1])
            ]
        elif node['opcode'] == 'extractelement' and len(args) == 2:
            vector = re.fullmatch(r'<\d+ x (.+)>', args[0][0])
            size = type_bytes(vector[1])
            index = int(args[1][1])
            raw = [part[index * size:(index + 1) * size] for part in operands[0]]
        elif binary_float:
            a, b = [numbers(t, data) for (t, _), data in zip(args, operands)]
            if a is not None and b is not None:
                out = [x + y if binary_float[1] == 'fadd' else x * y for x, y in zip(a, b)]
                if all(
                        math.isfinite(x) and abs(x) <= 3.4028234663852886e38
                        and struct.unpack('<f', struct.pack('<f', x))[0] == x for x in out):
                    raw = [tuple(struct.pack('<f', x)) for x in out]
        elif '@' + MMA + '(' in op and [t for t, _ in args] == ['<4 x half>', '<4 x half>', '<4 x float>']:
            a, b, c = [numbers(t, data) for (t, _), data in zip(args, operands)]
            if a is not None and b is not None and c is not None and len(set(a)) == len(set(b)) == 1:
                out = [16 * a[0] * b[0] + x for x in c]
                if all(
                        math.isfinite(x) and abs(x) <= 3.4028234663852886e38
                        and struct.unpack('<f', struct.pack('<f', x))[0] == x for x in out):
                    raw = [tuple(b''.join(struct.pack('<f', x) for x in out[i:i + 4])) for i in range(0, len(out), 4)]
        if result is not None and raw is not None:
            values[result] = raw
        binding = ('mma' if '@' + MMA +
                   '(' in op else 'fma' if '@llvm.fma.f32(' in op else 'fadd' if op.startswith('fadd float ') else
                   'fmul' if op.startswith('fmul float ') else 'perm' if '@llvm.mxc.byte.perm(' in op else None)
        if binding is None:
            continue
        expected = [
            '<4 x half>', '<4 x half>', '<4 x float>'
        ] if binding == 'mma' else ['float'] * (3 if binding == 'fma' else 2) if binding in ('fadd', 'fmul',
                                                                                             'fma') else ['i32'] * 3
        require([t for t, _ in args] == expected, 'primitive LLVM signature differs: ' + op)
        observed = [numbers(t, data) for (t, _), data in zip(args, operands)]
        conditions = dict(width_words=4 if binding == 'mma' else 1, mask='all', operand_binding=binding)
        if active_lanes is not None:
            conditions['active_lanes'] = active_lanes
        invariant_indices = (0, 1) if binding in ('mma', 'fma') else (1, )
        invariant = [observed[i] for i in invariant_indices]
        if all(v is not None and len(set(v)) == 1 for v in invariant) and len({v[0] for v in invariant}) == 1:
            conditions['invariant'] = invariant[0][0]
        restrictions = []
        if binding in ('fadd', 'fmul', 'fma', 'mma'):
            if all(v is not None and all(math.isfinite(x) for x in v) for v in observed):
                restrictions.append('finite exactly representable test inputs')
            if 'invariant' in conditions:
                restrictions.append('A/B=1 invariant' if binding in (
                    'mma', 'fma') and conditions['invariant'] == 1 else 'invariant RHS=' +
                                    str(float(conditions['invariant'])))
        elif binding == 'perm':
            if observed[2] is not None and len(set(observed[2])) == 1:
                conditions['selector'] = observed[2][0]
            if all(v is not None for v in observed) and active_lanes == 64:
                restrictions.append('full-wave integer bit patterns')
        evidence[node['id']] = dict(
            conditions=conditions, input_restrictions=restrictions,
            raw_role='C accumulator tuple' if binding in ('mma', 'fma') else 'first data operand feedback',
            operands=[dict(type=t, value=v) for t, v in args])
    return evidence


def selected_events(llvm, mlir, input_words=None, active_lanes=None):
    """Select LLVM operations, forward views, and retain shared-memory RAW."""
    nodes, begin, end = llvm_nodes(llvm)
    integers, pointers = address_values(nodes)
    scopes = operand_evidence(nodes, pointers, integers, input_words, active_lanes)
    accesses = {a['node']: a for a in memory_accesses(nodes, begin, end, pointers)}
    definitions = {n['result']: n for n in nodes if n['result']}
    pure_views = {
        'getelementptr', 'ptrtoint', 'inttoptr', 'bitcast', 'zext', 'sext', 'trunc', 'insertelement', 'extractelement',
        'insertvalue', 'extractvalue', 'freeze', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'lshr', 'ashr', 'icmp',
        'select'
    }
    live_values = set()
    dead = set()
    for node in reversed(nodes):
        if node['result'] and node['opcode'] in pure_views and node['result'] not in live_values:
            dead.add(node['id'])
        else:
            live_values.update(node['inputs'])
    address_nodes = set()
    address_inputs = []
    for node in nodes:
        if node['opcode'] == 'getelementptr':
            address_inputs.extend(node['inputs'])
        if node['id'] in accesses:
            address_inputs.extend(re.findall(r'ptr addrspace\(\d+\) (' + SSA + r')', node['llvm']))
            if accesses[node['id']]['form'].startswith('llvm.'):
                address_inputs.extend(re.findall(r', i64 (' + SSA + r')', node['llvm']))
    data_nodes = set()
    data_inputs = []
    for node in nodes[begin + 1:end]:
        access = accesses.get(node['id'])
        if node['opcode'] == 'store':
            data_inputs.extend(re.findall(SSA, node['llvm'].split(', ptr', 1)[0]))
        elif access is None and (node['opcode'] in ('fadd', 'fmul') or any(
                '@' + name + '(' in node['llvm']
                for name in (MMA, 'llvm.fma.f32', 'llvm.mxc.byte.perm', 'llvm.mxc.bsm.bpermute'))):
            data_inputs.extend(node['inputs'])
        elif access is not None and access['kind'] == 'global_store':
            payload = re.search(
                r'@llvm\.mxc\.stg\.predicator\.\w+\(ptr addrspace\(1\) [^,]+, i32 [^,]+, (' + TYPE + r') (' + SSA +
                r')', node['llvm'])
            if payload:
                data_inputs.append(payload[2])
    for pending, reached in ((address_inputs, address_nodes), (data_inputs, data_nodes)):
        while pending:
            definition = definitions.get(pending.pop())
            if definition is not None and definition['id'] not in reached:
                reached.add(definition['id'])
                if reached is not data_nodes or definition['id'] not in accesses:
                    pending.extend(definition['inputs'])
    writers = {}
    components = {}
    shared = {}
    events = []
    ignored = Counter()
    tags = source_tags(mlir)
    for node in nodes[begin + 1:end]:
        op = node['llvm']
        access = accesses.get(node['id'])
        deps = set().union(*(writers.get(x, set()) for x in node['inputs']))
        insertion = re.fullmatch(
            r'insertelement <(\d+) x [^>]+> (' + SSA + r'|undef|poison|zeroinitializer), '
            r'[^,]+ (' + SSA + r'|[^ ,]+), i\d+ (\d+)', op)
        extraction = re.fullmatch(r'extractelement <(\d+) x [^>]+> (' + SSA + r'), i\d+ (\d+)', op)
        if insertion:
            width, vector, scalar, index = insertion.groups()
            width, index = int(width), int(index)
            require(index < width, 'vector insertion index out of range')
            parts = list(components.get(vector, [writers.get(vector, set())] * width))
            parts[index] = writers.get(scalar, set())
            components[node['result']] = parts
            deps = set().union(*parts)
        elif extraction:
            width, vector, index = extraction.groups()
            require(int(index) < int(width), 'vector extraction index out of range')
            deps = components.get(vector, [writers.get(vector, set())] * int(width))[int(index)]
        elif node['opcode'] in ('insertelement', 'extractelement'):
            raise ValueError('unsupported vector component selection: ' + op)
        kind = native = None
        cells = set()
        if access is not None:
            kind = access['kind']
            bits = access['access_bytes'] * 8
            if kind.startswith('shared_'):
                cells = footprint(access)
                if kind == 'shared_load':
                    deps |= {shared[b] for b in cells if b in shared}
                native = ('lds_u16' if bits == 16 else f'lds_b{bits}') if kind == 'shared_load' else f'sts_b{bits}'
            else:
                require(kind in ('global_load', 'global_store'), 'unsupported memory space')
                native = f'ldg_{"u16" if bits == 16 else "b"+str(bits)}' if kind == 'global_load' else f'stg_b{bits}'
        elif f'@{MMA}(' in op:
            kind = 'mma'
            native = 'mma_16x16x16f16'
        elif '@llvm.mxc.barrier()' in op:
            kind = 'drain'
        elif 'arrive ' in op:
            kind = 'drain'
        elif op.startswith(('fadd float ', 'fmul float ')) or '@llvm.fma.f32(' in op:
            kind = 'arithmetic'
            native = 'add_f32' if op.startswith('fadd') else 'mul_f32' if op.startswith('fmul') else 'fmac_f32'
        elif '@llvm.mxc.byte.perm(' in op or '@llvm.mxc.bsm.bpermute(' in op:
            kind = 'perm'
            native = 'perm_b32' if '@llvm.mxc.byte.perm' in op else 'sm_bperm_b32'
        elif node['opcode'] == 'shufflevector':
            raise ValueError('unmapped shufflevector: requires an explicit rearrange annotation')
        if kind is None:
            scaffolding = node['opcode'] in {
                'getelementptr',
                'ptrtoint',
                'inttoptr',
                'bitcast',
                'zext',
                'sext',
                'trunc',
                'insertelement',
                'extractelement',
                'insertvalue',
                'extractvalue',
                'freeze',
                'fence',
            }
            if node['opcode'] in {'add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'lshr', 'ashr', 'icmp', 'select'}:
                require((node['id'] in address_nodes and node['id'] not in data_nodes) or node['id'] in dead,
                        'unpriced integer data operation: ' + op)
                scaffolding = True
            builtin = op in ('call i32 @llvm.mxc.thread.id.x()',
                             'call i64 @llvm.mxc.icmp.i64.i32(i32 1, i32 0, i32 34)')
            require(scaffolding or builtin, 'unsupported LLVM operation: ' + op)
            ignored[node['opcode']] += 1
            if node['result']: writers[node['result']] = deps
            continue
        source = None
        if kind in ('global_load', 'global_store', 'shared_load', 'shared_store', 'mma'):
            observed_kind, source = next(tags)
            require(kind == observed_kind, 'selected LLVM/MLIR operation order differs')
        event = {
            'id': node['id'], 'kind': kind, 'native': native, 'count': 1, 'members': [node['id']], 'deps': deps,
            'cells': cells, 'source': source, **scopes.get(node['id'], {})
        }
        if 'operands' in event:
            roles = (['A tuple', 'B tuple', 'C accumulator tuple'] if kind == 'mma' or native == 'fmac_f32' else
                     ['first data operand feedback', 'invariant RHS', 'selector'])
            event['operand_deps'] = {
                role: set().union(*(writers.get(x, set())
                                    for x in re.findall(SSA, operand['value'])))
                for role, operand in zip(roles, event['operands'])
            }
            event['operand_uses'] = [dict(node=node['id'], arguments=event['operands'])]
        if access is not None:
            event['memory'] = [access]
        # A group can charge count*I only while its members are independent.
        previous = events[-1] if events else None
        merge = (previous is not None and kind != 'drain' and previous['kind'] == kind and previous['native'] == native
                 and previous['source'] == source and not deps.intersection(previous['members'])
                 and previous.get('conditions') == event.get('conditions')
                 and previous.get('input_restrictions') == event.get('input_restrictions'))
        if merge:
            previous['count'] += 1
            previous['members'].append(node['id'])
            previous['deps'] = previous['deps'] | deps
            previous['cells'] = previous['cells'] | cells
            if 'memory' in event:
                previous['memory'].extend(event['memory'])
            for role, dependencies in event.get('operand_deps', {}).items():
                previous['operand_deps'][role] |= dependencies
            if 'operand_uses' in event:
                previous['operand_uses'].extend(event['operand_uses'])
        else:
            events.append(event)
        if node['result']: writers[node['result']] = {node['id']}
        if kind == 'shared_store':
            for byte in cells:
                shared[byte] = node['id']
    # The frozen stop instrumentation explicitly waits for output GVM before get_time.
    if 'arrive ' in nodes[end]['llvm'].split('get_time', 1)[0]:
        events.append(
            {'id': end, 'kind': 'drain', 'native': None, 'count': 1, 'members': [end], 'deps': set(), 'cells': set()})
    require(next(tags, None) is None, 'unconsumed selected MLIR operations')
    return events, dict(ignored)


def extract_events(llvm, ttgir, mlir, input_words=None, active_lanes=None):
    events, ignored = selected_events(llvm, mlir, input_words, active_lanes)
    combined = []
    i = 0
    while i < len(events):
        a = events[i]
        b = events[i + 1] if i + 1 < len(events) else None
        if (a['kind'] == 'shared_store' and b is not None and b['kind'] == 'shared_load' and a['source'] == b['source']
                and a['cells'] == b['cells'] and a['cells']):
            combined.append({
                'id': b['id'], 'kind': 'rearrange', 'native': None, 'count': 1, 'members': a['members'] + b['members'],
                'deps': a['deps'] | b['deps'], 'stores': a['count'], 'loads': b['count'], 'store_native': a['native'],
                'load_native': b['native'], 'memory': a.get('memory', []) + b.get('memory', []), 'source': a['source'],
                'exchange_bytes': len(a['cells'])
            })
            i += 2
        else:
            combined.append(a)
            i += 1
    locations = source_locations(ttgir)
    expected = Counter()
    for line in ttgir.splitlines():
        if '= ttg.convert_layout ' not in line: continue
        shape = re.search(r': tensor<([0-9x]+)(i\d+|f16|f32|f64),', line)
        loc = re.search(r'loc\((#loc\d*)\)', line)
        require(shape is not None and loc is not None, 'layout conversion shape/source is unavailable')
        bits = int(shape[2][1:])
        expected[locations[loc[1]]] += math.prod(map(int, shape[1].rstrip('x').split('x'))) * bits // 8
    observed = Counter()
    for event in combined:
        if event['kind'] == 'rearrange': observed[event['source']] += event['exchange_bytes']
    require(observed == expected, 'shared exchange bytes/source do not match TTGIR layout conversions')
    owners = {member: e['id'] for e in combined for member in e['members']}
    for e in combined:
        e['deps'] = sorted({owners.get(p, p) for p in e['deps']} - {e['id']})
        e.pop('cells', None)
        if 'operand_deps' in e:
            e['operand_deps'] = {
                role: sorted({owners.get(p, p)
                              for p in dependencies} - {e['id']})
                for role, dependencies in e['operand_deps'].items()
            }
    return combined, ignored
