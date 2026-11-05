"""Shared synthetic inputs and temporary, freshly qualified binaries."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catalog.query import load_catalog
from cost_model.llvm import MMA
from measurement.analyze import PROTOCOL
from measurement.benchmarks import lookup
from measurement.build import build, toolchain
from measurement.model import CostResult, write_json
from measurement.qualify import assembly_records, qualify


@pytest.fixture(scope='module')
def catalog():
    return load_catalog()


@pytest.fixture
def measurement_samples():

    def make(interval=4):
        return [{(n, phase): [4 if n == 0 else 8 + (n - 1) * interval] * 64
                 for n in PROTOCOL['counts']
                 for phase in (0, 8, 16, 24)}
                for _ in (1, 2)]

    return make


@pytest.fixture
def measurement_costs():

    def make(base='mma_f16_16x16x16', issue=4, latency=20):
        rows = {}
        for metric, value in (('independent_issue', issue), ('dependent_latency', latency)):
            if value is None:
                continue
            spec, _ = lookup(base + '.' + metric)
            rows[spec.measurement_id] = {
                **spec.to_dict(),
                'value': value,
                'unit': 'ticks/instruction',
                'native_opcode': spec.target.expected_isa,
                'timing_protocol': 'short_straight_line',
                'environment': {'target': 'xcore1000', 'gpu': 'MetaX C550'},
            }
        return rows

    return make


@pytest.fixture
def save_cost_result(measurement_costs):

    def save(root):
        out = Path(root) / 'add'
        out.mkdir()
        row = measurement_costs('fadd_f32', issue=4., latency=None)['fadd_f32.independent_issue']
        row = asdict(CostResult(run_id='run', **row))
        write_json(out / 'cost_result.json', row)
        write_json(
            out / 'qualification.json', {
                'passed': True, 'evidence': {
                    'native_opcode': 'add_f32',
                    'pure_target_window': True,
                    'matched_function_bytes': True,
                    'timing_protocol': 'short_straight_line',
                }
            })
        write_json(out / 'analysis.json', {'passed': True, 'interval_ticks': row['value'],
                                           'protocol': {'name': row['timing_protocol']}})
        write_json(out / 'run.json', {'measurement': {'measurement_id': row['measurement_id']},
                                      'timing_protocol': row['timing_protocol']})
        return out, row

    return save


@pytest.fixture(scope='module')
def qualified_runs():
    try:
        tools = toolchain()
    except (ValueError, FileNotFoundError):
        pytest.skip('MACA SDK unavailable')
    with tempfile.TemporaryDirectory(prefix='metax-qualification-') as tmp:
        runs = {}
        for mid in ('fadd_f32.dependent_latency', 'mma_f16_16x16x16.independent_issue', 'lds_b128.independent_issue',
                    'sts_b16.independent_issue', 'bpermute.i32.xor1.independent_issue',
                    'byte_perm.front.dependent_latency'):
            run = build(mid, Path(tmp) / mid, tools)
            result = qualify(run)
            assert result.passed, mid + ': ' + str(result.reason)
            runs[mid] = run
        yield runs


@pytest.fixture
def qualified_point(qualified_runs):

    def point(mid, n=3):
        spec, _ = lookup(mid)
        text = (qualified_runs[mid] / f'n{n}-p0/benchmark.s').read_text()
        return spec, assembly_records(text, 'bench')

    return point


@pytest.fixture
def llvm_kernel():

    def make(body):
        return ('define void @layout_gemm() {\n'
                '  %prime = call i64 asm sideeffect "get_time", "=s"()\n'
                '  %start = call i64 asm sideeffect "get_time", "=s"()\n' + body +
                '  %stop = call i64 asm sideeffect "get_time", "=s"()\n  ret void\n}')

    return make


@pytest.fixture
def mma_call():

    def make(result, c='c', a='a', b='b'):
        return f'  %{result} = call <4 x float> @{MMA}(<4 x half> %{a}, <4 x half> %{b}, <4 x float> %{c})\n'

    return make


@pytest.fixture
def make_nodes():

    def make(*operations):
        result = []
        for i, operation in enumerate(operations):
            target, op = operation.split(' = ', 1) if ' = ' in operation else (None, operation)
            result.append({'id': i, 'result': target, 'llvm': op})
        return result

    return make


@pytest.fixture
def issue_event():

    def make(node, kind='arithmetic', native='add_f32', count=1, deps=()):
        return {'id': node, 'kind': kind, 'native': native, 'count': count, 'deps': list(deps)}

    return make


@pytest.fixture
def layout_dataset(tmp_path, llvm_kernel, mma_call):
    import json

    kernels = tmp_path / 'kernels'
    kernels.mkdir()
    from cost_model.layouts import candidates, TARGET, COMPILE_OPTIONS
    names = [v['id'] for v in candidates()]
    operands = ('  %a = freeze <4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>\n'
                '  %b = freeze <4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>\n'
                '  %c = freeze <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>\n')
    llvm = llvm_kernel(operands + mma_call('x') + mma_call('y', 'x'))
    mlir = ('%start = llvm.inline_asm\n'
            '%x = llvm.call @llvm.mxc.mma.f32.16x16x16f16() loc(#loc1)\n'
            '%y = llvm.call @llvm.mxc.mma.f32.16x16x16f16() loc(#loc1)\n'
            '%stop = llvm.inline_asm\n#loc1 = loc("test.py":1:1)')
    for name in names:
        directory = kernels / name
        directory.mkdir()
        for file, text in (('compiler-input.ll', llvm), ('kernel.llir', llvm), ('kernel.ttgir', ''), ('kernel.mlir',
                                                                                                      mlir)):
            (directory / file).write_text(text)
        write_json(
            directory / 'run.json',
            dict(variant=next(v for v in candidates() if v['id'] == name), options=COMPILE_OPTIONS,
                 compiler=dict(target=TARGET), toolchain=dict(cpu='xcore1000'),
                 launch=dict(grid=[1, 1, 1], block=[64, 1, 1])))
    for process in (1, 2):
        directory = kernels / f'process-{process}'
        directory.mkdir()
        write_json(directory / 'environment.json', {'pid': process})
        write_json(directory / 'checks.json',
                   {n: {'before': {'passed': True}, 'after': {'passed': True}}
                    for n in names})
        with (directory / 'samples.jsonl').open('w') as stream:
            for i, name in enumerate(names):
                for repeat in range(64):
                    stream.write(
                        json.dumps({'variant': name, 'repeat': repeat, 'ticks': i + 100, 'start': 10, 'stop': i +
                                    110}) + '\n')
    return kernels, tmp_path / 'out'


def pytest_configure(config):
    config.addinivalue_line('markers', 'compilation: fresh native compiler/ELF regression')
