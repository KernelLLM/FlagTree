"""Reproduce measurement and within-shape TTGIR-layout ranking."""
from dataclasses import asdict
import os
import shutil
import tempfile

from measurement.build import output_path, toolchain
from measurement.measure import measure
from measurement.model import write_json
from measurement.qualify import require
from cost_model.layouts import (FORMAL_MEASUREMENTS, COMPILE_OPTIONS, TARGET, SAMPLING, candidates)
from cost_model.evidence import costs_from
from cost_model import model
from cost_model.execute import compile_candidate, sample, summarize


def run(output, device=0, sdk=None, mxcc=None, cost_root=None):
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'work'
    for name in ('tmp', 'cache', 'triton', 'torch'):
        (work / name).mkdir(parents=True)
    environment = dict(TMPDIR=str(work / 'tmp'), TMP=str(work / 'tmp'), TEMP=str(work / 'tmp'),
                       XDG_CACHE_HOME=str(work / 'cache'), TRITON_CACHE_DIR=str(work / 'triton'),
                       TORCH_EXTENSIONS_DIR=str(work / 'torch'), PYTHONDONTWRITEBYTECODE='1', TRITON_ALWAYS_COMPILE='1',
                       FLAGTREE_BACKEND='metax')
    previous = {key: os.environ.get(key) for key in (*environment, 'FLAGTREE_COMMON_IR', 'MACA_PATH')}
    previous_tempdir = tempfile.tempdir
    try:
        os.environ.update(environment)
        tempfile.tempdir = str(work / 'tmp')
        # CommonIR selects a different compiler pipeline.
        os.environ.pop('FLAGTREE_COMMON_IR', None)
        tools = toolchain(sdk, mxcc)
        os.environ['MACA_PATH'] = tools['sdk']
        measurements = output / 'measurements' if cost_root is None else cost_root
        if cost_root is None: measurements.mkdir()
        status = []
        for name in FORMAL_MEASUREMENTS if cost_root is None else ():
            result = measure(name, measurements / name, device, 0, tools)
            failed = isinstance(result, dict)
            status.append(dict(measurement=name, passed=not failed, result=result if failed else asdict(result)))
            write_json(output / 'measurement_status.json', status)
            print(name, 'FAILED: ' + result['reason'] if failed else f'{result.value} {result.unit}', flush=True)
        costs = costs_from(measurements)
        write_json(
            output / 'measurement_summary.json',
            dict(expected=len(FORMAL_MEASUREMENTS), published=len(costs),
                 failed=[r['measurement'] for r in status if not r['passed']]))
        kernels = output / 'layouts'
        kernels.mkdir()
        write_json(
            kernels / 'plan.json',
            dict(
                variants=candidates(), options=COMPILE_OPTIONS, target=TARGET,
                problem=dict(math='A@B+C', inputs='f16', accumulator_output='f32', global_storage='row-major',
                             tile='one complete MxN output tile; BLOCK_K=32; static K unroll'), sampling=SAMPLING,
                input_domain='ones for timing; ones and signed nonuniform integers for correctness',
                launch=dict(grid=[1, 1, 1], block=[64, 1, 1]), toolchain=tools))
        for variant in candidates():
            evidence = compile_candidate(variant, kernels / variant['id'], tools)
            print(variant['id'], 'ISA verified', evidence['timed_instruction_count'], 'instructions', flush=True)
        model.prepare(kernels, costs, output / 'ranking')
        # Correctness is checked before any timing in each worker, for every shape/layout.
        sample(kernels, tools, device)
        timing = summarize(kernels)
        evaluation = model.evaluate(kernels, output / 'ranking')
        summary = dict(published=len(costs), expected=len(FORMAL_MEASUREMENTS),
                       measurement_source='fresh run' if cost_root is None else str(cost_root),
                       measurement_passed=len(costs) == len(FORMAL_MEASUREMENTS), ranking_completed=True,
                       repeatability_passed=timing['repeatability_passed'], metrics=evaluation)
        write_json(output / 'summary.json', summary)
        require(summary['measurement_passed'], 'some formal measurements failed; ranking uses only fresh passing costs')
        require(summary['repeatability_passed'], 'GEMM process repeatability gate failed')
        return summary
    finally:
        tempfile.tempdir = previous_tempdir
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(work)
