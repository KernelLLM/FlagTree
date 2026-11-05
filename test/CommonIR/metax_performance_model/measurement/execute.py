"""Runtime ABI, numerical footprints, and one isolated sampling process."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import random
import time
from .build import command, ROOT
from .model import read_json, write_json
from .qualify import require, verify_qualification
from .benchmarks import lookup, TARGETS
from .benchmarks.llvm import WORDS, float_bits, seed, expected
from .analyze import protocol_for_run


def build_runtime(run):
    run = Path(run).resolve()
    sdk = read_json(run / 'run.json')['toolchain']['sdk']
    return command([
        'g++', '-shared', '-fPIC', '-std=c++17', ROOT / 'measurement/runtime.cpp', '-I' + sdk + '/include', '-I' + sdk +
        '/include/mcr', '-L' + sdk + '/lib', '-Wl,-rpath,' + sdk + '/lib', '-lmcruntime', '-o', run / 'runtime.so'
    ], run, 'runtime')


class Runtime:

    def __init__(self, run, device):
        self.run = Path(run)
        self.lib = C.CDLL(str(self.run / 'runtime.so'))
        l = self.lib
        l.error_text.argtypes = [C.c_int]
        l.error_text.restype = C.c_char_p
        l.device_info.argtypes = [C.c_int, C.c_void_p, C.c_uint]
        l.load_module.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
        l.unload_module.argtypes = [C.c_void_p]
        l.allocate.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        l.release.argtypes = [C.c_void_p]
        l.launch.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t]
        info = C.create_string_buffer(4096)
        self.check(l.device_info(device, info, len(info)))
        self.environment = json.loads(info.value)
        require(self.environment['name'] in ('MetaX C550', 'MXC550') and self.environment['warp_size'] == 64,
                'requires C550/wave64')
        self.input = C.c_void_p()
        self.output = C.c_void_p()
        self.modules = {}
        self.points = {}
        try:
            self.check(l.allocate(C.byref(self.input), WORDS * 4))
            self.check(l.allocate(C.byref(self.output), WORDS * 4))
            saved = read_json(self.run / 'run.json')
            self.spec, _ = lookup(saved['measurement']['measurement_id'])
            for case in saved['cases']:
                key = (case['n'], case['pad'])
                directory = self.run / case['path']
                self.modules[key] = self.load(directory / 'benchmark.fatbin')
                self.points[key] = read_json(directory / 'qualification.json')
        except Exception:
            self.close()
            raise

    def load(self, path):
        image = C.create_string_buffer(path.read_bytes())
        module = C.c_void_p()
        self.check(self.lib.load_module(image, C.byref(module)))
        return module

    def check(self, code):
        if code: raise RuntimeError(self.lib.error_text(code).decode())

    def launch(self, key, inp):
        out = (C.c_uint32 * WORDS)(*[0xa5a5a5a5] * WORDS)
        self.check(self.lib.launch(self.modules[key], b'bench', self.input, self.output, inp, out, WORDS * 4))
        check_output(self.spec, key[0], inp, out)
        clocks = []
        for lane in range(64):

            def u64(i):
                return int(out[lane * 64 + i]) | int(out[lane * 64 + i + 1]) << 32

            clocks.append(tuple(u64(i) for i in (0, 2, 4, 6)))
        require(len(set(clocks)) == 1, 'wave counters/PC differ')
        ticks, start, stop, pc = clocks[0]
        require(start > 0 and stop >= start and ticks == stop - start, 'invalid counter sample')
        q = self.points[key]
        base = pc - q['pc_offset'] - 8
        begin = base + q['start_offset']
        end = base + q['stop_offset']
        require(begin // 4096 == (end + 7) // 4096, 'actual GPU window crosses 4 KiB')
        require(all(v == 0xa5a5a5a5 for v in list(out)[4096:]), 'output extent differs')
        return {
            'n': key[0], 'pad': key[1], 'ticks': ticks, 'start': start, 'stop': stop, 'start_mod32': begin % 32,
            'stop_mod32': end % 32, 'function_gpu_address': base, 'start_gpu_address': begin, 'stop_gpu_address': end
        }

    def close(self):
        for module in self.modules.values():
            self.lib.unload_module(module)
        self.modules.clear()
        if self.input.value:
            self.lib.release(self.input)
            self.input.value = None
        if self.output.value:
            self.lib.release(self.output)
            self.output.value = None


def inputs(spec, pattern):
    t = TARGETS[spec.target.name]
    kind = t['kind']
    floating = kind in ('fadd', 'fmul', 'fma', 'mma')
    buf = (C.c_uint32 * WORDS)()
    if 'address_terms' in t:
        for i in range(WORDS):
            buf[i] = (0x12345678 ^ i * 0x10101 ^ pattern * 0xa5a5a5a5) & 0xffffffff
    for lane in range(64):
        for c in range(spec.independent_chains):
            for j in range(t['width']):
                ix = lane * 64 + c * t['width'] + j
                value = seed(lane, c, j, pattern, floating)
                buf[ix] = value
                buf[ix + 32] = (value ^ 0xa5a5a5a5) & 0xffffffff
    if kind == 'mma':
        for ix in (4096, 4097, 4100, 4101):
            buf[ix] = 0x3c003c00
    elif floating:
        buf[4096] = float_bits(t['rhs'])
        buf[4100] = float_bits(1.)
    elif kind in ('integer', 'perm'):
        buf[4096] = t['rhs']
    return buf


def check_output(spec, n, inp, out):
    t = TARGETS[spec.target.name]
    for lane in range(64):
        for c in range(spec.independent_chains):
            for j in range(t['width']):
                want = expected(spec, n, inp, lane, c, j)
                actual = int(out[lane * 64 + 16 + c * t['width'] + j])
                require(actual == want,
                        f'correctness failed n={n} lane={lane} slot={c} component={j}: {actual:08x} != {want:08x}')


def correctness(runtime):
    for pattern in range(3):
        inp = inputs(runtime.spec, pattern)
        for key in runtime.modules:
            runtime.launch(key, inp)
    return {
        'passed': True, 'patterns': 3, 'binaries': len(runtime.modules), 'active_lanes': 64, 'complete_result_tuples':
        True
    }


def sample(run, process):
    run = Path(run)
    saved = verify_qualification(run)
    plan = read_json(run / 'sampling.json')
    protocol = protocol_for_run(saved)
    require(plan['protocol'] == protocol, 'sampling protocol differs')
    out = run / f'process-{process}'
    out.mkdir()
    rt = Runtime(run, plan['device'])
    try:
        require(rt.environment['uuid'] == plan['environment']['uuid'], 'sample GPU differs')
        before = correctness(rt)
        write_json(out / 'checks.json', {'before': before, 'process_id': os.getpid(), 'environment': rt.environment})
        (run / f'ready-{process}').touch()
        deadline = time.monotonic() + 300
        while not (run / f'sample-start-{process}').exists():
            require(time.monotonic() < deadline, 'sampling barrier timeout')
            time.sleep(.03)
        inp = inputs(rt.spec, 0)
        keys = list(rt.modules)
        for key in keys:
            for _ in range(protocol['warmup']):
                rt.launch(key, inp)
        rng = random.Random(plan['seed'] + process)
        with (out / 'samples.jsonl').open('x') as f:
            for repeat in range(protocol['samples']):
                rng.shuffle(keys)
                for key in keys:
                    f.write(json.dumps({'repeat': repeat, **rt.launch(key, inp)}) + '\n')
        after = correctness(rt)
        write_json(out / 'checks.json',
                   {'before': before, 'after': after, 'process_id': os.getpid(), 'environment': rt.environment})
    finally:
        rt.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--process', type=int, choices=(1, 2), required=True)
    a = p.parse_args()
    sample(a.run, a.process)
