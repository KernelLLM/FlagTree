"""Ranking metrics and read-only evaluation of explicitly supplied kernel samples."""
import json
import math
from pathlib import Path
import statistics

from measurement.model import read_json
from measurement.qualify import require
from .layouts import SAMPLING


def ranks(values):
    ranks = [0.] * len(values)
    for value in set(values):
        indices = [i for i, x in enumerate(values) if x == value]
        rank = sum(x < value for x in values) + (len(indices) + 1) / 2
        for i in indices:
            ranks[i] = rank
    return ranks


def metrics(names, predicted, timings, oracle_names=None):
    """Rank the predicted subset; compare selections with the full supplied oracle."""
    oracle_names = list(names if oracle_names is None else oracle_names)
    require(set(names) <= set(oracle_names), 'oracle must include every predicted candidate')
    actual = sorted(oracle_names, key=lambda n: (timings[n], n))
    measured = [timings[n] for n in names]
    pairs = [(i, j) for i in range(len(names)) for j in range(i) if measured[i] != measured[j]]
    score = sum(.5 if predicted[i] == predicted[j] else float((predicted[i] - predicted[j]) *
                                                              (measured[i] - measured[j]) > 0) for i, j in pairs)
    pr, mr = ranks(predicted), ranks(measured)
    pm, mm = statistics.mean(pr), statistics.mean(mr)
    denom = math.sqrt(sum((x - pm)**2 for x in pr) * sum((y - mm)**2 for y in mr))
    first = [i for i, x in enumerate(predicted) if x == min(predicted)]
    best = timings[actual[0]]
    result = {
        'pairwise_ranking_accuracy':
        score / len(pairs) if pairs else None, 'spearman_rank':
        sum((x - pm) * (y - mm) for x, y in zip(pr, mr)) / denom if denom else None, 'top1_tie_group_size':
        len(first), 'top1_tie_regret_range_percent': [(min(measured[i] for i in first) / best - 1) * 100,
                                                      (max(measured[i] for i in first) / best - 1) * 100],
        'top_k_coverage': {
            str(k):
            len(
                set(actual[:k])
                & {names[i]
                   for i, x in enumerate(predicted)
                   if x <= sorted(predicted)[min(k, len(names)) - 1]}) / min(k, len(actual))
            for k in (1, 3, 5)
        }
    }
    picked = sorted(range(len(names)), key=lambda i: (predicted[i], names[i]))
    result['deterministic_top1'] = names[picked[0]]
    result['top1_regret_percent'] = (measured[picked[0]] / best - 1) * 100
    result['exact_k_coverage'] = {
        str(k): len({names[i] for i in picked[:k]} & set(actual[:k])) / min(k, len(actual))
        for k in (1, 3, 5)
    }
    result['latency_mape_percent'] = statistics.mean(abs(v / m - 1) * 100 for v, m in zip(predicted, measured))
    return result


def measured_latencies(kernels, names):
    samples = {n: [[], []] for n in names}
    pids = []
    for p in (1, 2):
        root = Path(kernels) / f'process-{p}'
        checks = read_json(root / 'checks.json')
        pids.append(read_json(root / 'environment.json')['pid'])
        seen = set()
        for name in names:
            require(checks[name]['before']['passed'] and checks[name]['after']['passed'],
                    'frozen correctness incomplete')
        for line in (root / 'samples.jsonl').open():
            row = json.loads(line)
            key = (row['variant'], row['repeat'])
            require(key not in seen and key[0] in names and key[1] in range(64), 'invalid frozen sample')
            require(row['ticks'] > 0 and row['ticks'] == row['stop'] - row['start'], 'invalid frozen counter')
            seen.add(key)
            samples[key[0]][p - 1].append(row['ticks'])
        require(len(seen) == len(names) * 64, 'frozen sample coverage')
    require(len(set(pids)) == 2, 'frozen independent processes missing')
    measured = {n: statistics.median(samples[n][0] + samples[n][1]) for n in names}
    process_medians = {n: list(map(statistics.median, samples[n])) for n in names}
    for name, medians in process_medians.items():
        spread = abs(medians[0] - medians[1]) / statistics.mean(medians) * 100
        require(spread <= SAMPLING['repeatability_percent'],
                f'{name}: process repeatability gate failed ({spread:.2f}%)')
    return measured, process_medians
