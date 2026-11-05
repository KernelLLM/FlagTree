"""Fresh compilation and two-process execution of explicit TTGIR layouts."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import statistics
import struct
import subprocess
import sys
import time

from measurement.build import ROOT, command
from measurement.model import read_json, write_json
from measurement.qualify import assembly_records, device_elf, functions, require
from .evidence import counter_window, isa_graph
from .layouts import COMPILE_OPTIONS, TARGET, SAMPLING, WORDS, inputs_for, apply_layout, buffer_layout
from .ranking import measured_latencies


def compile_candidate(variant, out, tools):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton._common_ir import ENABLED
    from triton._C.libtriton import metax
    from triton._C.libtriton import ir
    from triton.backends.metax.compiler import MACABackend
    from .kernels import layout_gemm

    out.mkdir()
    m, n, k = variant['shape']
    constexprs = dict(M=m, N=n, K=k, BLOCK_K=variant['block_k'])
    source = ASTSource(layout_gemm, {'Input': '*i32', 'Output': '*i32', **{k: 'constexpr'
                                                                           for k in constexprs}}, constexprs=constexprs)
    native = metax.translate_llvmir_to_mcfatbin
    frontend = MACABackend.make_ttgir

    def select_layout(module, metadata, options, capability):
        module = frontend(module, metadata, options, capability)
        text = str(module)
        (out / 'frontend.ttgir').write_text(text)
        path = out / 'selected.ttgir'
        path.write_text(apply_layout(text, variant))
        selected = ir.parse_mlir_module(str(path), module.context)
        selected.context = module.context
        return selected

    def capture(llvm, compiler, sdk, extra_options):
        (out / 'compiler-input.ll').write_text(llvm)
        write_json(out / 'compiler_call.json', dict(compiler=compiler, sdk=sdk, extra_options=extra_options))
        return native(llvm, compiler, sdk, extra_options)

    metax.translate_llvmir_to_mcfatbin = capture
    MACABackend.make_ttgir = staticmethod(select_layout)
    try:
        kernel = triton.compile(source, target=GPUTarget(**TARGET), options=COMPILE_OPTIONS)
    finally:
        metax.translate_llvmir_to_mcfatbin = native
        MACABackend.make_ttgir = staticmethod(frontend)
    for stage, artifact in kernel.asm.items():
        if stage == 'mcfatbin': (out / 'benchmark.fatbin').write_bytes(artifact)
        elif isinstance(artifact, str): (out / ('kernel.' + stage)).write_text(artifact)
    require((out / 'compiler-input.ll').read_text() == kernel.asm['llir'], 'compiler LLVM differs')
    require(kernel.metadata.global_scratch_size == kernel.metadata.profile_scratch_size == 0,
            'unexpected scratch allocation')
    write_json(
        out / 'run.json',
        dict(
            variant=variant, toolchain=tools, options=COMPILE_OPTIONS, name=kernel.name, shared=kernel.metadata.shared,
            k_iterations=k // variant['block_k'], expected_mma=m * n * k // 4096,
            encodings=[line for line in kernel.asm['ttgir'].splitlines() if line.startswith('#') and '#ttg.' in line],
            launch=dict(grid=[1, 1, 1],
                        block=[64, 1, 1]), compiler=dict(entry='triton.compile(ASTSource)', module=triton.__file__,
                                                         common_ir_enabled=ENABLED, target=TARGET)))
    return offline_check(out, tools, kernel.name)


def offline_check(out, tools, symbol):
    flags = shlex.split(read_json(out / 'compiler_call.json')['extra_options'])
    command([
        tools['mxcc'], '-cc1', '-O2', '-triple', tools['triple'], '-target-cpu', tools['cpu'], *flags, '-x', 'ir', '-S',
        out / 'kernel.llir', '-o', out / 'kernel.s'
    ], out, 'assembly')
    command([
        tools['mxcc'], '-cc1as', '-triple', tools['triple'], '-target-cpu', tools['cpu'], '-filetype', 'obj',
        out / 'kernel.s', '-o', out / 'reassembled.o'
    ], out, 'assemble')
    (out / 'device.elf').write_bytes(device_elf((out / 'benchmark.fatbin').read_bytes()))
    address, actual = functions(out / 'device.elf')[symbol]
    _, assembled = functions(out / 'reassembled.o')[symbol]
    require(actual == assembled[:len(actual)], 'final ELF and independent object differ')
    records = assembly_records((out / 'kernel.s').read_text(), symbol)
    require(len(actual) == len(records) * 8 and records[-1]['op'] == 'endk', 'ISA/function extent differs')
    clocks = counter_window(records)
    timed = records[clocks[1] + 1:clocks[2]]
    require(not any(r['op'].startswith(('bra', 'br_', 'call', 'ret')) for r in timed), 'unexpected native control flow')
    require(not any(r['op'] in ('and_xmsk', 'or_xmsk', 'set_xmsk') for r in records), 'execution mask changes')
    counts = Counter(r['op'] for r in timed)
    llvm_mma = sum('@llvm.mxc.mma.f32.16x16x16f16(' in line and 'call ' in line
                   for line in (out / 'kernel.llir').read_text().splitlines())
    expected_mma = read_json(out / 'run.json')['expected_mma']
    require(counts['mma_16x16x16f16'] == llvm_mma == expected_mma, 'LLVM/ISA GEMM MMA count differs')
    resources = {
        key: int(match[1])
        for key in ('mtreg_count', 'streg_count', 'private_memory_size', 'share_memory_size')
        if (match := re.search(r'\.' + key + r':\s*(\d+)', (out / 'kernel.s').read_text()))
    }
    resources['dynamic_shared_bytes'] = read_json(out / 'run.json')['shared']
    from .llvm import selected_events
    events, ignored = selected_events((out / 'kernel.llir').read_text(), (out / 'kernel.mlir').read_text())
    selected = Counter()
    for event in events:
        if event['native']: selected[event['native']] += event['count']
    mapped_ops = set(selected) | {
        op for op in counts if op.startswith(('ldg_', 'stg_', 'lds_', 'sts_', 'mma_'))
    }
    mapping = {
        op: dict(llvm_count=selected.get(op, 0), isa_count=counts.get(op, 0),
                 verified=selected.get(op, 0) == counts.get(op, 0)) for op in sorted(mapped_ops)
    }
    mismatches = [op for op, row in mapping.items() if not row['verified']]
    evidence = dict(passed=not mismatches, independent_object_matches=True, function_bytes=len(actual),
                    function_address=address,
                    counts=dict(counts), timed_instruction_count=len(timed), llvm_mma_count=llvm_mma,
                    waits=[dict(index=i, clauses=r['args'])
                           for i, r in enumerate(timed)
                           if r['op'] == 'arrive'], native_dependency_graph=isa_graph(records), resources=resources,
                    llvm_to_isa=mapping, ignored_llvm=ignored,
                    unmapped_native_counts={op: count
                                            for op, count in counts.items()
                                            if op not in selected})
    write_json(out / 'offline.json', evidence)
    require(not mismatches, 'LLVM/ISA primitive counts differ: ' + ', '.join(mismatches))
    return evidence


class LayoutRuntime:

    def __init__(self, root, device, variants=None):
        self.root = Path(root)
        self.lib = C.CDLL(str(self.root / 'runtime.so'))
        l = self.lib
        l.error_text.argtypes = [C.c_int]
        l.error_text.restype = C.c_char_p
        l.device_info.argtypes = [C.c_int, C.c_void_p, C.c_uint]
        l.allocate.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        l.release.argtypes = [C.c_void_p]
        l.load_module.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
        l.unload_module.argtypes = [C.c_void_p]
        l.launch_layout_gemm.argtypes = [C.c_void_p] * 5 + [C.c_size_t, C.c_uint]
        self.input = C.c_void_p()
        self.output = C.c_void_p()
        self.modules = {}
        self.shared = {}
        try:
            info = C.create_string_buffer(4096)
            self.check(l.device_info(device, info, len(info)))
            self.environment = json.loads(info.value)
            require(self.environment['name'] in ('MetaX C550', 'MXC550') and self.environment['warp_size'] == 64,
                    'requires C550/wave64')
            self.check(l.allocate(C.byref(self.input), WORDS * 4))
            self.check(l.allocate(C.byref(self.output), WORDS * 4))
            self.variants = variants if variants is not None else read_json(self.root / 'plan.json')['variants']
            self.configurations = {v['id']: v for v in self.variants}
            for variant in self.variants:
                name = variant['id']
                image = C.create_string_buffer((self.root / name / 'benchmark.fatbin').read_bytes())
                module = C.c_void_p()
                self.check(l.load_module(image, C.byref(module)))
                self.modules[name] = module
                self.shared[name] = read_json(self.root / name / 'run.json')['shared']
        except Exception:
            self.close()
            raise

    def check(self, code):
        if code: raise RuntimeError(self.lib.error_text(code).decode())

    def invoke(self, name, inputs):
        output = (C.c_uint32 * WORDS)()
        self.check(
            self.lib.launch_layout_gemm(self.modules[name], self.input, self.output, inputs, output, WORDS * 4,
                                        self.shared[name]))

        def values(base):
            return [int(output[base + 2 * i]) | int(output[base + 2 * i + 1]) << 32 for i in range(64)]

        offsets = buffer_layout(self.configurations[name])
        ticks, starts, stops = (values(offsets[key]) for key in ('ticks', 'start', 'stop'))
        require(all(len(set(v)) == 1 for v in (ticks, starts, stops)), 'lane counter values differ')
        require(ticks[0] == (stops[0] - starts[0]) % (1 << 64) and 0 < ticks[0] < 1 << 32, 'invalid counter interval')
        return output, dict(ticks=ticks[0], start=starts[0], stop=stops[0])

    def close(self):
        for module in self.modules.values():
            self.lib.unload_module(module)
        self.modules.clear()
        for ptr in (self.input, self.output):
            if ptr.value:
                self.lib.release(ptr)
                ptr.value = None


def correctness(runtime, variant):
    m, n, kdim = variant['shape']
    max_error = 0.
    for pattern in (0, 1):
        inputs, a, b, c = inputs_for(variant, pattern)
        output, _ = runtime.invoke(variant['id'], inputs)
        for i in range(m):
            for j in range(n):
                expected = c[i][j] + sum(a[i][k] * b[k][j] for k in range(kdim))
                actual = struct.unpack('<f', struct.pack('<I', output[i * n + j]))[0]
                error = abs(actual - expected)
                max_error = max(max_error, error)
                require(
                    math.isfinite(actual) and error <= (0 if pattern == 0 else 1e-4),
                    f"{variant['id']}: output[{i},{j}] {actual} != {expected}")
    return dict(passed=True, patterns=2, elements=m * n, active_lanes=64, ones_atol=0, nonuniform_atol=1e-4, rtol=0,
                max_absolute_error=max_error)


def worker(root, process, device):
    destination = root / f'process-{process}'
    destination.mkdir()
    runtime = LayoutRuntime(root, device)
    try:
        variants = runtime.variants
        checks = {v['id']: {'before': correctness(runtime, v)} for v in variants}
        write_json(destination / 'checks.json', checks)
        write_json(destination / 'environment.json', dict(pid=os.getpid(), **runtime.environment))
        (destination / 'ready').touch()
        deadline = time.monotonic() + SAMPLING['timeout_seconds']
        while not (destination / 'start').exists():
            require(time.monotonic() < deadline, 'sampling barrier timed out')
            time.sleep(.05)
        inputs = {v['id']: inputs_for(v, 0)[0] for v in variants}
        for name, inp in inputs.items():
            for _ in range(SAMPLING['warmup']):
                runtime.invoke(name, inp)
        rng = random.Random(SAMPLING['shuffle_seed'] + process)
        names = list(inputs)
        with (destination / 'samples.jsonl').open('x') as stream:
            for repeat in range(SAMPLING['samples']):
                order = list(names)
                rng.shuffle(order)
                for name in order:
                    _, counter = runtime.invoke(name, inputs[name])
                    stream.write(json.dumps(dict(variant=name, repeat=repeat, **counter)) + '\n')
        for v in variants:
            checks[v['id']]['after'] = correctness(runtime, v)
        write_json(destination / 'checks.json', checks)
    finally:
        runtime.close()


def sample(root, tools, device):
    command([
        'g++', '-shared', '-fPIC', '-std=c++17',
        Path(__file__).with_name('runtime.cpp'), '-I' + tools['sdk'] + '/include', '-I' + tools['sdk'] + '/include/mcr',
        '-L' + tools['sdk'] + '/lib', '-Wl,-rpath,' + tools['sdk'] + '/lib', '-lmcruntime', '-o', root / 'runtime.so'
    ], root, 'runtime')
    children = []
    streams = []
    try:
        for process in (1, 2):
            stream = (root / f'process-{process}.log').open('x')
            streams.append(stream)
            children.append(
                subprocess.Popen([
                    sys.executable, '-B', '-m', 'cost_model.execute', '--worker',
                    str(process), '--output',
                    str(root), '--device',
                    str(device)
                ], cwd=root, env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=stream,
                                 stderr=subprocess.STDOUT))
        deadline = time.monotonic() + SAMPLING['timeout_seconds']
        while not all((root / f'process-{p}/ready').exists() for p in (1, 2)):
            require(all(c.poll() is None for c in children), 'GEMM correctness failed; see process logs')
            require(time.monotonic() < deadline, 'GEMM correctness timed out')
            time.sleep(.05)
        for process, child in enumerate(children, 1):
            (root / f'process-{process}/start').touch()
            require(child.wait(timeout=SAMPLING['timeout_seconds']) == 0, 'GEMM sampling failed; see process logs')
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
        for process in (1, 2):
            for name in ('start', 'ready'):
                (root / f'process-{process}' / name).unlink(missing_ok=True)


def summarize(root):
    names = [v['id'] for v in read_json(root / 'plan.json')['variants']]
    measured, medians = measured_latencies(root, names)
    rows = []
    for name in names:
        spread = abs(medians[name][0] - medians[name][1]) / statistics.mean(medians[name]) * 100
        rows.append(
            dict(id=name, measured_ticks=measured[name], process_medians=medians[name], process_spread_percent=spread,
                 repeatability_passed=spread <= SAMPLING['repeatability_percent']))
    summary = dict(variants=rows, repeatability_passed=all(r['repeatability_passed'] for r in rows),
                   fitted_parameters={})
    write_json(root / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(1, 2), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    worker(args.output, args.worker, args.device)
