"""Build, qualify, sample in two independent processes, then publish an interval."""
from dataclasses import asdict
from pathlib import Path
import os
import subprocess
import sys
import time
from .build import ROOT, build, output_path
from .model import CostResult, write_json
from .qualify import qualify, require, verify_qualification
from .execute import build_runtime, Runtime
from .analyze import analyze, protocol_for_run
from .benchmarks import lookup


def qualify_measurement(measurement_id, output_dir, toolchain=None):
    return qualify(build(measurement_id, output_dir, toolchain))


def sample_qualified(out, device=0, seed=0):
    out = Path(out).resolve()
    saved = verify_qualification(out)
    protocol = protocol_for_run(saved)
    require(not (out / 'sampling.json').exists(), 'sampling already started')
    spec, _ = lookup(saved['measurement']['measurement_id'])
    children = []
    streams = []
    try:
        runtime_command = build_runtime(out)
        rt = Runtime(out, device)
        try:
            environment = rt.environment
        finally:
            rt.close()
        write_json(
            out / 'sampling.json', {
                'protocol': protocol, 'device': device, 'seed': seed, 'environment': environment, 'runtime_build':
                runtime_command, 'visibility': {
                    name: os.environ.get(name)
                    for name in ('MACA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                 'GPU_DEVICE_ORDINAL')
                }, 'launch': {'grid': [1, 1, 1], 'block': [64, 1, 1], 'active_lanes': 64}
            })
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT)}
        for p in (1, 2):
            stream = (out / f'process-{p}.log').open('x')
            streams.append(stream)
            children.append(
                subprocess.Popen(
                    [sys.executable, '-B', '-m', 'measurement.execute', '--run',
                     str(out), '--process',
                     str(p)], cwd=out, env=env, stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 300
        while not all((out / f'ready-{p}').exists() for p in (1, 2)):
            require(all(c.poll() is None for c in children), 'correctness child failed; see process logs')
            require(time.monotonic() < deadline, 'correctness timeout')
            time.sleep(.03)
        for p, child in enumerate(children, 1):
            (out / f'sample-start-{p}').touch()
            require(child.wait(timeout=300) == 0, 'sampling child failed; see process log')
        analysis = analyze(out)
        write_json(out / 'analysis.json', analysis)
        require(analysis['passed'], 'interval stability failed: ' + ', '.join(analysis['failure_reasons']))
        result = CostResult(
            out.name, spec.measurement_id, asdict(spec.target), spec.metric, spec.raw_role, spec.independent_chains,
            spec.input_restrictions, analysis['interval_ticks'], 'ticks/instruction', {
                **environment, 'gpu': environment['name'], 'target': saved['toolchain']['cpu'], 'toolchain':
                saved['toolchain']['version'].splitlines()[0]
            }, spec.target.expected_isa, spec.conditions, 'short_straight_line')
        write_json(out / 'cost_result.json', asdict(result))
        return result
    finally:
        for child in children:
            if child.poll() is None: child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in streams:
            stream.close()
        for name in ('ready-1', 'ready-2', 'sample-start-1', 'sample-start-2'):
            (out / name).unlink(missing_ok=True)


def measure(measurement_id, output_dir, device=0, seed=0, toolchain=None):
    out = output_path(output_dir)
    lookup(measurement_id)
    if out.exists(): raise FileExistsError(out)
    try:
        q = qualify_measurement(measurement_id, out, toolchain)
        require(q.passed, q.reason)
        return sample_qualified(out, device, seed)
    except Exception as error:
        if out.is_dir(): write_json(out / 'failure.json', {'reason': str(error)})
        return {'failed': True, 'measurement_id': measurement_id, 'reason': str(error)}
