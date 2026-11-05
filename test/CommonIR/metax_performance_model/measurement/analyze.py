"""Cross-count, process and PC-phase interval validation."""
from pathlib import Path
from statistics import median
import json
from .model import read_json, write_json
from .qualify import require, verify_qualification
from .benchmarks.llvm import COUNTS, window_counts
from .benchmarks import lookup
from .build import output_path

PROTOCOL = {
    'name': 'short_straight_line', 'counts': list(COUNTS), 'padding': [0, 1, 2, 3], 'processes': 2, 'samples': 64,
    'warmup': 8, 'counter_unit': 'ticks'
}


def protocol_for_run(saved):
    spec, _ = lookup(saved['measurement']['measurement_id'])
    counts = list(window_counts(spec))
    require(saved['timing_protocol'] == PROTOCOL['name'] and saved['counts'] == counts, 'sampling protocol differs')
    return {**PROTOCOL, 'counts': counts}


def summarize(processes, counts=COUNTS):
    failures = []
    medians = []
    intervals = []
    spans = []
    anchors = []
    overheads = []
    for p in processes:
        med = {key: median(v) for key, v in p.items()}
        medians.append({f'{n}:{phase}': v for (n, phase), v in sorted(med.items())})
        if any(min(v) != max(v) for v in p.values()): failures.append('samples not repeatable within a binary')
        require(
            set(med) == {(n, phase)
                         for n in counts
                         for phase in (0, 8, 16, 24)}, 'actual PC/count coverage incomplete')
        for phase in (0, 8, 16, 24):
            overheads.append(med[0, phase])
            anchors.append(med[1, phase])
            intervals.extend(med[n, phase] - med[n - 1, phase] for n in range(2, min(max(counts), 6) + 1))
            if 8 in counts: spans.append(med[8, phase] - med[6, phase])
    value = intervals[0]
    if value <= 0 or len(set(intervals)) != 1 or any(x != 2 * value for x in spans):
        failures.append('no common steady-state interval across counts/processes/phases')
    if len(set(anchors)) != 1 or len(set(overheads)) != 1:
        failures.append('timer/first-target window depends on process/phase')
    return {
        'passed': not failures, 'failure_reasons': sorted(set(failures)), 'interval_ticks':
        value if not failures else None, 'process_medians_ticks': medians, 'unit_increments_ticks': intervals,
        'two_target_increments_ticks': spans, 'timer_overhead_ticks': overheads, 'first_target_window_ticks': anchors,
        'interpretation': 'consecutive issue/RAW-chain interval; not arbitrary-consumer completion'
    }


def analyze(run_dir, new_output=None):
    run = Path(run_dir).resolve()
    saved = verify_qualification(run)
    protocol = protocol_for_run(saved)
    sampling = read_json(run / 'sampling.json')
    require(sampling['protocol'] == protocol, 'sampling protocol differs')
    processes = []
    pids = []
    for p in (1, 2):
        directory = run / f'process-{p}'
        checks = read_json(directory / 'checks.json')
        require(checks['before']['passed'] and checks['after']['passed'], 'correctness failed/incomplete')
        require(checks['environment']['uuid'] == sampling['environment']['uuid'], 'sample device differs')
        groups = {}
        seen = set()
        addresses = {}
        for line in (directory / 'samples.jsonl').open():
            row = json.loads(line)
            key = (row['n'], row['pad'], row['repeat'])
            require(
                key not in seen and row['n'] in protocol['counts'] and row['pad'] in range(4)
                and row['repeat'] in range(64), 'invalid/duplicate sample point')
            require(row['ticks'] == row['stop'] - row['start'] and row['start'] > 0, 'invalid counter sample')
            seen.add(key)
            groups.setdefault((row['n'], row['start_mod32']), []).append(row['ticks'])
            addresses.setdefault((row['n'], row['pad']), set()).add(row['function_gpu_address'])
        require(seen == {(n, pad, r)
                         for n in protocol['counts']
                         for pad in range(4)
                         for r in range(64)}, 'sample coverage incomplete')
        require(all(len(v) == 1 for v in addresses.values()), 'function address changed')
        processes.append(groups)
        pids.append(checks['process_id'])
    require(len(set(pids)) == 2, 'independent processes required')
    result = {**summarize(processes, protocol['counts']), 'protocol': protocol, 'process_ids': pids}
    if new_output:
        out = output_path(new_output)
        require(not out.exists(), 'analysis output exists')
        write_json(out, result)
    return result
