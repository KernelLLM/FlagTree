"""Cost loading and offline native dependency evidence; no latency prediction."""
from pathlib import Path
import re
from measurement.model import read_json, load_cost_result
from measurement.qualify import require


def counter_window(records):
    clocks = [i for i, r in enumerate(records) if r['op'] == 'get_time']
    require(len(clocks) == 3, 'prime/start/stop counter structure differs')
    require(
        all(i + 1 < len(records) and records[i + 1]['op'] == 'arrive' and records[i + 1]['args'] == ['slcnt(0)']
            for i in clocks), 'counter result is consumed without a scalar completion wait')
    require(records[clocks[-1] - 1]['op'] == 'arrive' and records[clocks[-1] - 1]['args'] == ['gvmcnt(0)'],
            'output completion must precede stop')
    return clocks


def costs_from(root):
    """Load qualified primitive costs; incomplete measurements stay model gaps."""
    from dataclasses import asdict
    costs = {}
    for path in sorted(Path(root).rglob('cost_result.json')):
        try:
            row = asdict(load_cost_result(path))
        except (TypeError, ValueError) as error:
            raise ValueError(f'invalid primitive CostResult: {path}: {error}') from error
        q = read_json(path.parent / 'qualification.json')
        require(
            q['passed'] and q['evidence']['native_opcode'] == row['native_opcode']
            and q['evidence']['pure_target_window'] and q['evidence']['matched_function_bytes']
            and q['evidence']['timing_protocol'] == 'short_straight_line'
            and row['timing_protocol'] == 'short_straight_line', f'primitive qualification missing: {path}')
        analysis = read_json(path.parent / 'analysis.json')
        require(analysis['passed'], f'measurement gate failed: {path}')
        require(analysis.get('interval_ticks') == row['value'], f'CostResult differs from analyzed interval: {path}')
        run = read_json(path.parent / 'run.json')
        require(run['measurement']['measurement_id'] == row['measurement_id'],
                f'CostResult measurement differs from run: {path}')
        require(run['timing_protocol'] == row['timing_protocol'] == analysis.get('protocol', {}).get('name'),
                f'CostResult timing protocol differs from analysis/run: {path}')
        require(row['measurement_id'] not in costs, 'duplicate measurement: ' + row['measurement_id'])
        costs[row['measurement_id']] = {**row, 'path': str(path)}
    return costs


def reg_bits(token, width=1):
    match = re.fullmatch(r'([rs])(\d+)(?:_(lo|hi))?', token.strip())
    if not match:
        return []
    kind, base, half = match.groups()
    return [f'{kind}{int(base)+i}:{h}' for i in range(width) for h in ((half, ) if half else ('lo', 'hi'))]


def instruction_io(row):
    op, args = row['op'], row['args']
    if op in ('arrive', 'barrier', 'endk', 'snop'):
        return [], [], True
    if op.startswith('mma_'):
        widths = (4, 2, 2, 4)
    elif op == 'pk_fma_f32':
        widths = (2, 2, 2, 2)
    else:
        width = 4 if '128' in op else 2 if '64' in op or op == 'get_time' else 1
        widths = (width, ) + (1, ) * (len(args) - 1)
        if op.startswith(('ldg_', 'ldu_')) and len(args) > 1:
            widths = (width, 2) + widths[2:]
        if op.startswith('stg_'):
            widths = (2, 1, 1, width)
        if op.startswith('sts_'):
            widths = (1, 1, width)
        if op in ('shl_b64', 'shr_b64'):
            widths = (2, 1, 2)
        if op == 'smov_b64':
            widths = (2, 2)
    known = (op.startswith(('mma_', 'ldg_', 'stg_', 'lds_', 'sts_', 'ldu_')) or op in {
        'pk_fma_f32', 'add_f32', 'mul_f32', 'fmac_f32', 'get_time', 'mov_b32', 'smov_b32', 'smov_b64', 'rfl_b32',
        'shl_b32', 'shr_b32', 'shl_b64', 'shr_b64', 'add_i32', 'add_u32', 'add_u16', 'and_b32', 'or_b32', 'xor_b32',
        'and_or_b32', 'and_xor_b32', 'or3_b32', 'shl_or_b32', 'bfe_i32', 'bfe_u32', 'sm_bperm_b32', 'cmp_gt_u32',
        'cmp_eq_u32', 'cmp_lt_u32', 'csel_b32', 'sadd_co_u32', 'sadd_co_i32', 'saddc_co_u32', 'ssub_co_u32',
        'ssubb_co_u32', 'xadd_u32', 'perm_b32'
    })
    if not known:
        return [], [], False
    tokens = [re.findall(r'\b[rs]\d+(?:_(?:lo|hi))?\b', arg) for arg in args]
    registers = [
        sum((reg_bits(t, widths[i] if i < len(widths) else 1) for t in part), []) for i, part in enumerate(tokens)
    ]
    store = op.startswith(('stg_', 'sts_'))
    writes = [] if store else registers[0] if registers else []
    reads = sum(registers if store else registers[1:], [])
    if op == 'fmac_f32':
        reads += registers[0]
    if op in ('smov_b32', 'smov_b64'):
        if args and args[0] == 'md': writes.append('md')
        if len(args) > 1 and args[1] == 'md': reads.append('md')
    if op in ('saddc_co_u32', 'ssubb_co_u32'):
        reads.append('scalar_carry')
    if op in ('sadd_co_u32', 'sadd_co_i32', 'saddc_co_u32', 'ssub_co_u32', 'ssubb_co_u32'):
        writes.append('scalar_carry')
    return reads, writes, True


def wait_clauses(text):
    parts = [x.strip() for x in text.split('&')]
    clauses = []
    for part in parts:
        match = re.fullmatch(r'(gvmcnt|bsmcnt|slcnt)\((\d+)\)', part)
        require(match is not None, 'unsupported wait: ' + text)
        clauses.append((match[1], int(match[2])))
    require(len({q for q, _ in clauses}) == len(clauses), 'duplicate wait queue: ' + text)
    return clauses


def pending_queues(op, args=()):
    if op.startswith(('sts_', 'lds_')) or op == 'sm_bperm_b32': return ('bsmcnt', )
    if op.startswith(('ldg_', 'stg_')):
        return ('gvmcnt', 'bsmcnt') if any(re.search(r'\bbsm\b', a) for a in args) else ('gvmcnt', )
    if op.startswith('ldu_') or op == 'get_time': return ('slcnt', )
    return ()


def isa_graph(records):
    writers = {}
    pending = {'gvmcnt': [], 'bsmcnt': [], 'slcnt': []}
    graph = []
    for i, row in enumerate(records):
        reads, writes, supported = instruction_io(row)
        if not supported:
            # Unknown writes must not leave stale definitions looking like proven RAW.
            writers.clear()
        async_copy = row['op'].startswith('ldg_') and 'bsmcnt' in pending_queues(row['op'], row['args'])
        if async_copy:
            # ldg.bsm writes shared memory through MD, not its printed first operand as a register result.
            reads = list(
                dict.fromkeys(reads + sum((reg_bits(r)
                                           for a in row['args']
                                           for r in re.findall(r'\b[rs]\d+\b', a)), [])))
            reads.append('md')
            writes = []
        edges = [{'from': writers[r], 'kind': 'register_RAW', 'component': r} for r in reads if r in writers]
        wait = None
        if row['op'] == 'arrive':
            clauses = []
            for queue, threshold in wait_clauses(' & '.join(row['args'])):
                clauses.append({'queue': queue, 'threshold': threshold, 'candidates': list(pending[queue])})
                edges.extend({'from': j, 'kind': 'completion' if threshold == 0 else 'may_completion', 'queue': queue}
                             for j in pending[queue])
                if threshold == 0: pending[queue].clear()
            wait = {'clauses': clauses}
        graph.append({
            'id': i, **row, 'reads': reads, 'writes': writes, 'edges': edges, 'wait': wait, 'queues':
            list(pending_queues(row['op'], row['args'])), 'register_analysis_supported': supported
        })
        for r in writes:
            writers[r] = i
        for queue in pending_queues(row['op'], row['args']):
            pending[queue].append(i)
    return graph
