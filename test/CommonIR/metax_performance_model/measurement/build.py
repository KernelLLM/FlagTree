"""Build controlled LLVM with the selected MACA toolchain."""
from pathlib import Path
import os
import subprocess
from .model import write_json
from .benchmarks import lookup
from .benchmarks.llvm import PHASES, window_counts

ROOT = Path(__file__).resolve().parents[1]


def output_path(path):
    path = Path(path).resolve()
    protected = [ROOT]
    for parent in ROOT.parents:
        if (parent / '.git').exists():
            protected.append(parent)
            break
    if any(path == p or path.is_relative_to(p) for p in protected):
        raise ValueError('output must be outside source trees')
    return path


def toolchain(sdk=None, mxcc=None):
    sdk = Path(sdk or os.environ.get('MACA_PATH', '/opt/maca')).resolve()
    compiler = Path(mxcc or sdk / 'mxgpu_llvm/bin/mxcc').resolve()
    if not compiler.is_file(): raise ValueError('mxcc not found: ' + str(compiler))
    return {
        'sdk': str(sdk), 'mxcc': str(compiler), 'version': subprocess.check_output([str(compiler), '--version'],
                                                                                   text=True), 'cpu': 'xcore1000',
        'triple': 'mxc-metax-macahca'
    }


def command(argv, out, label):
    result = subprocess.run(list(map(str, argv)), cwd=out, capture_output=True, text=True)
    (out / (label + '.log')).write_text(result.stdout + result.stderr)
    if result.returncode: raise RuntimeError(label + ' failed; see ' + str(out / (label + '.log')))
    return {'argv': list(map(str, argv)), 'returncode': result.returncode}


def build(measurement_id, output, tools=None):
    out = output_path(output)
    spec, family = lookup(measurement_id)
    out.mkdir(parents=True, exist_ok=False)
    tools = tools or toolchain()
    cases = []
    counts = window_counts(spec)
    for n in counts:
        for pad in PHASES:
            point = out / f'n{n}-p{pad}'
            point.mkdir()
            ll = point / 'benchmark.ll'
            ll.write_text(family.render(spec, n, pad))
            commands = []
            for label, args in [('binary', [
                    '-O2', '-fatbin', '-maca-link', '-input-is-device', '--maca-path=' + tools['sdk'],
                    '-offload-arch=' + tools['cpu'], '-x', 'ir', ll, '-o', point / 'benchmark.fatbin'
            ]),
                                ('assembly', [
                                    '-cc1', '-O2', '-triple', tools['triple'], '-target-cpu', tools['cpu'], '-x', 'ir',
                                    '-S', ll, '-o', point / 'benchmark.s'
                                ]),
                                ('reassemble', [
                                    '-cc1as', '-triple', tools['triple'], '-target-cpu', tools['cpu'], '-filetype',
                                    'obj', point / 'benchmark.s', '-o', point / 'reassembled.o'
                                ])]:
                commands.append(command([tools['mxcc'], *args], point, label))
            cases.append({'n': n, 'pad': pad, 'path': point.name, 'commands': commands})
    write_json(
        out / 'run.json', {
            'measurement': spec.to_dict(), 'toolchain': tools, 'counts': list(counts), 'cases': cases,
            'timing_protocol': 'short_straight_line'
        })
    return out
