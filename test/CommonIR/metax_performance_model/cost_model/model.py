"""MetaX Cost Model configuration, static prediction and ranking evaluation."""
from collections import Counter
from pathlib import Path
import argparse
import csv

from measurement.build import output_path
from measurement.model import read_json, write_json, validate_cost_result
from measurement.qualify import require
from .evidence import costs_from
from .layouts import candidates, inputs_for, TARGET, COMPILE_OPTIONS
from .llvm import extract_events
from .ranking import measured_latencies, metrics, ranks
from .timeline import predict

MEMORY_D = dict(lds_d=64, sts_d=32, gvm_d=256)


def cost_table(costs):
    """Keep measured primitive costs separate from provisional memory assumptions."""
    rates = {}
    for name, row in costs.items():
        validate_cost_result(row)
        if row['native_opcode'].startswith(('lds_', 'sts_')):
            continue
        require(row['metric'] in ('independent_issue', 'dependent_latency'), 'unsupported primitive metric: ' + name)
        rate = rates.setdefault(
            row['native_opcode'],
            dict(source='measured MetaX primitive; grouped operation cost', measurement_ids=[], measurements=[]))
        rate['measurement_ids'].append(name)
        rate['measurements'].append({
            field: row[field]
            for field in ('measurement_id', 'metric', 'value', 'conditions', 'raw_role', 'independent_chains',
                          'input_restrictions')
        })
    rates['shared_load'] = dict(I=4, D=None, source='provisional b32 shared-load representative', measurement_ids=[])
    rates['shared_store'] = dict(I=costs['sts_b32.independent_issue']['value'], D=None,
                                 source='measured b32 shared-store representative',
                                 measurement_ids=['sts_b32.independent_issue'])
    for kind in ('global_load', 'global_store'):
        rates[kind] = dict(I=costs['fadd_f32.independent_issue']['value'], D=None,
                           source='provisional global issue = measured FADD issue quantum',
                           measurement_ids=['fadd_f32.independent_issue'])
    return dict(unit='ticks', rates=rates, memory_d=dict(MEMORY_D))


def prepare(kernels, costs, output):
    """Save LLVM-only predictions before reading any kernel timing."""
    kernels = Path(kernels)
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    table = cost_table(costs)
    predictions = {}
    features = {}
    ignored = {}
    variants = read_json(kernels / 'plan.json')['variants'] if (kernels / 'plan.json').exists() else candidates()
    for variant in sorted(variants, key=lambda v: v['id']):
        name = variant['id']
        directory = kernels / name
        llvm = (directory / 'compiler-input.ll').read_text()
        require(llvm == (directory / 'kernel.llir').read_text(), 'compiler LLVM inputs differ')
        run = read_json(directory / 'run.json')
        require(
            run['variant'] == variant and run['options'] == COMPILE_OPTIONS and run['compiler']['target'] == TARGET
            and run['toolchain']['cpu'] == 'xcore1000' and run['launch'] == dict(grid=[1, 1, 1], block=[64, 1, 1]),
            'cost scope requires the frozen C550/wave64 launch')
        try:
            events, ignored[name] = extract_events(llvm, (directory / 'kernel.ttgir').read_text(),
                                                   (directory / 'kernel.mlir').read_text(),
                                                   input_words=inputs_for(variant, 0)[0], active_lanes=64)
            predictions[name] = predict(events, table['rates'], table['memory_d'])
            counts = Counter()
            for event in events:
                if event['native']: counts[event['native']] += event['count']
                elif event['kind'] == 'rearrange':
                    counts[event['store_native']] += event['stores']
                    counts[event['load_native']] += event['loads']
            features[name] = dict(
                shape=variant['shape'], layout=variant['layout'], events=len(events), primitive_counts=dict(counts),
                raw_edges=sum(len(e['deps']) for e in events), k_iterations=variant['shape'][2] // variant['block_k'],
                layout_conversions=(directory / 'kernel.ttgir').read_text().count('= ttg.convert_layout '),
                rearranges=sum(e['kind'] == 'rearrange' for e in events), coverage=predictions[name]['coverage'])
            write_json(directory / 'events.json', events)
        except ValueError as error:
            predictions[name] = dict(predicted_cost=None, model_gap=str(error), coverage=None)
            features[name] = dict(model_gap=str(error))
    write_json(
        output / 'plan.json',
        dict(
            cost_table=table, timing_used_for_parameters=False,
            prediction_input='LLVM SSA and TTGIR/MLIR annotations; no final ISA counts',
            input_scope='inputs_for(variant, 0): ones; operand restrictions are proved for this input only',
            wait_rule='drain outstanding work; no FIFO, thresholds or fixed wait service',
            ignored='address arithmetic, views and fence direct cost is zero; data ALU requires a cost',
            rearrange='drain + shared-store completion + shared-load issue', limitations=[
                'provisional memory readiness is not qualified latency',
                'bank conflicts, cache and code-layout effects are not modeled'
            ]))
    write_json(output / 'features.json', features)
    write_json(output / 'ignored_llvm.json', ignored)
    write_json(output / 'predictions.json', dict(unit=table['unit'], candidates=predictions, timing_read=False))
    return predictions


def evaluate(kernels, output):
    output = Path(output)
    saved = read_json(output / 'predictions.json')
    variants = read_json(Path(kernels) / 'plan.json')['variants'] if (Path(kernels) /
                                                                      'plan.json').exists() else candidates()
    all_names = [v['id'] for v in variants]
    measured, process_medians = measured_latencies(kernels, all_names)
    evaluation = {}
    rows = []
    for shape in sorted({tuple(v['shape']) for v in variants}):
        shape_names = [v['id'] for v in variants if tuple(v['shape']) == shape]
        names = [name for name in shape_names if saved['candidates'][name]['predicted_cost'] is not None]
        values = [saved['candidates'][n]['predicted_cost'] for n in names]
        key = 'x'.join(map(str, shape))
        evaluation[key] = dict(metrics=metrics(names, values, measured, oracle_names=shape_names) if names else None,
                               predicted=len(names), total=len(shape_names),
                               prediction_coverage=len(names) / len(shape_names), ranking_scope='predicted_subset',
                               oracle_scope='all_shape_candidates')
        rows.extend(
            dict(candidate=n, shape=key, predicted_cost=v, unit=saved['unit'], rank=rank, measured_ticks=measured[n],
                 error_percent=(v / measured[n] - 1) * 100) for n, v, rank in zip(names, values, ranks(values)))
    fields = ('candidate', 'shape', 'predicted_cost', 'unit', 'rank', 'measured_ticks', 'error_percent')
    with (output / 'ranking.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'evaluation.json',
               dict(shapes=evaluation, measured_ticks=measured, process_medians=process_medians, fitted_parameters=[]))
    return evaluation


def run(kernels, cost_root, output):
    prepare(kernels, costs_from(cost_root), output)
    return evaluate(kernels, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels', type=Path, required=True)
    parser.add_argument('--costs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.kernels, args.costs, args.output))
