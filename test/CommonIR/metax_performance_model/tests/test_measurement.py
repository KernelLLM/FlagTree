"""Benchmark generation, native qualification, correctness and interval publication."""
import ctypes
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from measurement.analyze import PROTOCOL, protocol_for_run, summarize
from measurement.benchmarks import TARGETS, list_measurements, lookup
from measurement.benchmarks.llvm import COUNTS, WORDS, expected, render
from measurement.build import ROOT, output_path
from measurement.execute import check_output, inputs
from measurement.measure import measure
from measurement.model import QualificationResult, read_json
from measurement.qualify import device_elf, functions, qualify, verify_point


class TestGeneration:

    def test_scope_and_metrics(self):
        specs = list_measurements()
        assert len(specs) == 44
        assert len({s.measurement_id for s in specs}) == 44
        for spec in specs:
            assert spec.conditions['active_lanes'] == 64
            assert spec.metric in ('independent_issue', 'dependent_latency')
            assert spec.independent_chains == ((4 if spec.target.name == 'bpermute.i32.xor1' else 8)
                                               if spec.metric == 'independent_issue' else 1)
            text = render(spec, min(8, spec.independent_chains) if spec.metric == 'independent_issue' else 8, 0)
            assert text.count('call i64 @llvm.readcyclecounter()') == 2
            assert 'br label' not in text

    def test_bit_exact_oracles_reject_bad_last_lane(self):
        for spec in list_measurements():
            inp = inputs(spec, 1)
            out = (ctypes.c_uint32 * WORDS)()
            w = TARGETS[spec.target.name]['width']
            for lane in range(64):
                for c in range(spec.independent_chains):
                    for j in range(w):
                        out[lane * 64 + 16 + c * w + j] = expected(spec, 8, inp, lane, c, j)
            check_output(spec, 8, inp, out)
            out[63 * 64 + 16] ^= 1
            with pytest.raises(ValueError, match='lane=63'):
                check_output(spec, 8, inp, out)

    def test_protocol_and_output_boundary(self):
        assert COUNTS == (0, 1, 2, 3, 4, 5, 6, 8)
        with pytest.raises(ValueError):
            output_path(ROOT / 'generated')
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / 'link'
            link.symlink_to(ROOT, target_is_directory=True)
            with pytest.raises(ValueError):
                output_path(link / 'artifact')
        with pytest.raises(ValueError):
            lookup('lds_b32.dependent_latency')
        with pytest.raises(ValueError):
            lookup('async_copy.i32.fixed_md.completion')


@pytest.mark.compilation
class TestQualification:

    def test_all_native_windows_and_boundaries(self, qualified_runs):
        for run in qualified_runs.values():
            q = read_json(run / 'qualification.json')
            assert q['evidence']['points'] == (20 if 'bpermute' in run.name else 32)
            assert q['evidence']['pure_target_window']
            assert q['evidence']['matched_function_bytes']

    def test_reject_raw_opcode_and_extra_wait(self, qualified_point):
        mid = 'fadd_f32.dependent_latency'
        spec, rows = qualified_point(mid)
        first = next((r for r in rows if r['op'] == 'add_f32'))
        first['args'][2] = 'r999'
        with pytest.raises(ValueError, match='RAW'):
            verify_point(spec, 3, rows)
        for opcode in ('arrive', 'snop', 'mul_f32'):
            spec, rows = qualified_point(mid)
            next((r for r in rows if r['op'] == 'add_f32'))['op'] = opcode
            with pytest.raises(ValueError):
                verify_point(spec, 3, rows)

    def test_reject_shared_address_selector_and_tuple_alias(self, qualified_point):
        for mid, opcode, slot in [('lds_b128.independent_issue', 'lds_b128', 1),
                                  ('byte_perm.front.dependent_latency', 'perm_b32', 3),
                                  ('mma_f16_16x16x16.independent_issue', 'mma_16x16x16f16', 1)]:
            spec, rows = qualified_point(mid)
            next((r for r in rows if r['op'] == opcode))['args'][slot] = 'r999'
            with pytest.raises(ValueError):
                verify_point(spec, 3, rows)

    def test_elf_function_byte_rejection(self, qualified_runs):
        run = qualified_runs['fadd_f32.dependent_latency']
        p = run / 'n3-p0/reassembled.o'
        raw = p.read_bytes()
        body = functions(p)['bench'][1]
        offset = raw.index(body)
        changed = bytearray(raw)
        changed[offset] ^= 1
        p.write_bytes(changed)
        try:
            assert 'function bytes differ' in qualify(run).reason
        finally:
            p.write_bytes(raw)
        assert qualify(run).passed

    def test_page_rejection(self, qualified_runs):
        run = qualified_runs['fadd_f32.dependent_latency']
        real = functions

        def address(path):
            value = real(path)
            if Path(path).name == 'device.elf':
                code = value['bench'][1]
                q = read_json(run / 'n0-p0/qualification.json')
                value['bench'] = (4096 - 8 - q['start_offset'], code)
            return value

        with patch('measurement.qualify.functions', side_effect=address):
            assert not qualify(run).passed
        assert qualify(run).passed

    def test_invalid_bundle(self):
        with pytest.raises(ValueError):
            device_elf(b'not an ELF bundle')


class TestMeasurement:

    def test_interval_not_short_window_ratio(self, measurement_samples):
        r = summarize(measurement_samples(16))
        assert r['passed']
        assert r['interval_ticks'] == 16
        assert set(r['timer_overhead_ticks']) == {4}
        assert 'fit' not in r

    def test_reject_nonlinear_process_and_layout_drift(self, measurement_samples):
        for key in ((3, 0), (8, 16), (1, 24), (0, 8)):
            data = measurement_samples()
            data[1][key] = [300] * 64
            assert not summarize(data)['passed']
        data = measurement_samples()
        data[0][4, 0][-1] += 4
        assert not summarize(data)['passed']

    def test_protocol_rejects_other_timing(self):
        assert protocol_for_run({
            'timing_protocol': 'short_straight_line', 'counts': PROTOCOL['counts'], 'measurement':
            {'measurement_id': 'fadd_f32.independent_issue'}
        }) == PROTOCOL
        with pytest.raises(ValueError):
            protocol_for_run({
                'timing_protocol': 'slope', 'counts': [0, 32, 64, 128], 'measurement':
                {'measurement_id': 'fadd_f32.independent_issue'}
            })

    def test_failed_qualification_never_enters_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'

            def built(*args):
                out.mkdir()
                return out

            with patch('measurement.measure.build',
                       side_effect=built), patch('measurement.measure.qualify', return_value=QualificationResult(
                           False, 'wrong RAW', {})), patch('measurement.measure.build_runtime') as runtime:
                r = measure('fadd_f32.dependent_latency', out)
            assert r['failed']
            runtime.assert_not_called()
            assert not (out / 'cost_result.json').exists()
            with pytest.raises(FileExistsError):
                measure('fadd_f32.dependent_latency', out)
