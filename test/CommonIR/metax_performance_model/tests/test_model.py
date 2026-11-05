"""MetaX Cost Model, primitive costs, LLVM dependencies and catalog queries."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import yaml

from catalog.query import FIELDS, load_catalog, query_catalog, table
from cost_model.evidence import costs_from, counter_window, isa_graph, pending_queues, reg_bits, wait_clauses
from cost_model.layouts import candidates, inputs_for
from cost_model.llvm import address_values, extract_events, llvm_nodes, memory_accesses, memory_form
from cost_model.model import MEMORY_D, cost_table, prepare, evaluate
from cost_model.ranking import metrics, ranks
from cost_model.timeline import Timeline, predict
from measurement.benchmarks import lookup
from measurement.benchmarks.shared import GEMM_LDS_MAPS, lane_offsets
from measurement.model import write_json
from measurement.qualify import numeric_step


class TestConfiguration:

    def test_frozen_layout_compile_and_sampling_settings(self):
        from cost_model.layouts import COMPILE_OPTIONS, TARGET, SAMPLING, FORMAL_MEASUREMENTS
        from measurement.benchmarks import list_measurements
        from cost_model.layouts import SHAPES, LAYOUTS, apply_layout
        variants = candidates()
        assert len(variants) == len(SHAPES) * len(LAYOUTS) == 24
        assert len({v['id'] for v in variants}) == 24
        for shape in SHAPES:
            group = [v for v in variants if v['shape'] == list(shape)]
            assert len(group) == 6
            assert {v['block_k'] for v in group} == {32}
            assert {v['a_layout'] for v in group} == {v['b_layout'] for v in group} == {'row'}
        text = (
            '#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [1, 1], order = [1, 0]}>\n'
            '#mma = #ttg.maca_mma<{elementsMNK = [1, 1, 4], colMajor = 0}>\n'
            '#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>\n'
            '%b = ttg.local_load %mem : !ttg.memdesc<32x32xf16, #shared, #smem> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma}>>\n'
            'tt.dot %a, %b, %c')
        assert len({apply_layout(text, v) for v in variants}) == 6
        assert all(apply_layout(text, v).endswith('tt.dot %a, %b, %c') for v in variants)
        assert COMPILE_OPTIONS == dict(num_warps=1, num_stages=1, enable_fp_fusion=False, pipeline='basic')
        assert TARGET == dict(backend='maca', arch=80, warp_size=64)
        assert SAMPLING == dict(processes=2, warmup=8, samples=64, shuffle_seed=1700, repeatability_percent=5,
                                timeout_seconds=300)
        excluded = {
            'i32_and.independent_issue', 'i32_and.dependent_latency', 'i32_or.independent_issue',
            'lds_b32.independent_issue', 'bpermute.i32.xor1.independent_issue', 'lds_gemm.map1.independent_issue',
            'lds_gemm.map0.independent_issue', 'lds_gemm.map4.independent_issue'
        }
        assert len(FORMAL_MEASUREMENTS) == len(set(FORMAL_MEASUREMENTS)) == 36
        assert set(FORMAL_MEASUREMENTS) == {s.measurement_id for s in list_measurements()} - excluded
        assert MEMORY_D == dict(lds_d=64, sts_d=32, gvm_d=256)

    def test_measured_rates_and_provisional_assumptions_stay_separate(self, measurement_costs, tmp_path):
        costs = {
            **measurement_costs('fadd_f32', 4, 4), **measurement_costs('sts_b32', 8, None),
            **measurement_costs('lds_b128', 4, None)
        }
        table = cost_table(costs)
        rates = table['rates']
        assert rates['shared_load'] == dict(I=4, D=None, source='provisional b32 shared-load representative',
                                            measurement_ids=[])
        assert rates['shared_store']['I'] == 8
        assert rates['global_load']['I'] == 4 and rates['global_load']['D'] is None
        assert 'lds_b128' not in rates and 'sts_b32' not in rates
        assert 'modes' not in table and 'grid' not in table['memory_d']
        assert costs['lds_b128.independent_issue']['value'] == 4
        write_json(tmp_path / 'model_config.json', table)
        assert costs_from(tmp_path) == {}

    def test_predictions_are_saved_before_timing_and_need_no_elf(self, layout_dataset, measurement_costs):
        kernels, out = layout_dataset
        costs = {
            **measurement_costs('fadd_f32', 4, 4), **measurement_costs('sts_b32', 8, None),
            **measurement_costs('mma_f16_16x16x16', 4, 20)
        }
        with patch('measurement.qualify.functions',side_effect=AssertionError('prediction must not read ELF')), \
             patch('cost_model.model.measured_latencies',side_effect=AssertionError('prediction must not read timings')):
            prepare(kernels, costs, out)
        saved = json.loads((out / 'predictions.json').read_text())
        assert saved['timing_read'] is False
        assert 'modes' not in saved
        assert all(v['predicted_cost'] == 24 for v in saved['candidates'].values())
        features = json.loads((out / 'features.json').read_text())
        assert all(v['primitive_counts']['mma_16x16x16f16'] == 2 for v in features.values())
        result = evaluate(kernels, out)
        assert all(v['metrics']['pairwise_ranking_accuracy'] == .5 for v in result.values())
        assert 'mode' not in (out / 'ranking.tsv').read_text().splitlines()[0]
        assert not list(out.glob('*sensitivity*'))
        (kernels / 'process-1/samples.jsonl').unlink()
        with pytest.raises(FileNotFoundError):
            evaluate(kernels, out)

    def test_evaluation_keeps_unpredicted_candidates_in_oracle(self, layout_dataset):
        kernels, out = layout_dataset
        variants = candidates()
        missing = variants[0]['id']
        abstained_shape = variants[6]['shape']
        out.mkdir()
        write_json(out / 'predictions.json', dict(unit='ticks', candidates={
            v['id']: dict(predicted_cost=None if v['id'] == missing or v['shape'] == abstained_shape else 1)
            for v in variants
        }))
        result = evaluate(kernels, out)
        partial = result['16x32x32']
        assert partial['predicted'] == 5 and partial['total'] == 6
        assert partial['prediction_coverage'] == 5 / 6
        assert partial['ranking_scope'] == 'predicted_subset'
        assert partial['oracle_scope'] == 'all_shape_candidates'
        assert partial['metrics']['top1_regret_percent'] > 0
        assert partial['metrics']['exact_k_coverage']['1'] == 0
        abstained = result['32x32x64']
        assert abstained['prediction_coverage'] == 0
        assert abstained['metrics'] is None

    def test_reproduction_pipeline_only_calls_one_predictor(self, monkeypatch, tmp_path):
        import reproduce
        import tempfile
        events = []
        monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
        for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                     'PYTHONDONTWRITEBYTECODE', 'TRITON_ALWAYS_COMPILE', 'FLAGTREE_BACKEND', 'MACA_PATH'):
            monkeypatch.setenv(name, 'unused')
        monkeypatch.setattr(reproduce, 'FORMAL_MEASUREMENTS', ('fadd_f32.independent_issue', ))
        monkeypatch.setattr(reproduce, 'toolchain', lambda *a: {'sdk': '/sdk'})
        monkeypatch.setattr(reproduce, 'measure',
                            lambda *a: events.append('measure') or {'failed': True, 'reason': 'gate failed'})
        monkeypatch.setattr(reproduce, 'costs_from', lambda *a: {})
        monkeypatch.setattr(reproduce, 'compile_candidate',
                            lambda *a: events.append('compile') or {'timed_instruction_count': 16})
        monkeypatch.setattr(reproduce.model, 'prepare', lambda *a: events.append('predict'))
        monkeypatch.setattr(reproduce, 'sample', lambda *a: events.append('timing'))
        monkeypatch.setattr(reproduce, 'summarize', lambda *a: {'repeatability_passed': True})
        monkeypatch.setattr(reproduce.model, 'evaluate',
                            lambda *a: events.append('evaluate') or {'pairwise_ranking_accuracy': .5})
        out = tmp_path / 'fresh'
        with pytest.raises(ValueError, match='formal measurements failed'):
            reproduce.run(out)
        assert events == ['measure'] + ['compile'] * len(candidates()) + ['predict', 'timing', 'evaluate']
        assert json.loads((out / 'summary.json').read_text())['metrics'] == {'pairwise_ranking_accuracy': .5}
        assert not (out / 'work').exists()
        with pytest.raises(FileExistsError):
            reproduce.run(out)

    def test_shared_lane_maps_match_measurement_definitions(self):
        for index, (width, terms) in enumerate(GEMM_LDS_MAPS):
            offsets = lane_offsets(terms)
            occupied = [p + j for p in offsets for j in range(width)]
            assert len(occupied) == len(set(occupied))
            spec, _ = lookup(f'lds_gemm.map{index}.independent_issue')
            assert spec.conditions['lane_byte_offsets'] == offsets

    def test_layout_correctness_checks_signed_inputs_and_all_outputs(self):
        import ctypes as C
        import struct
        from cost_model.execute import correctness

        class Runtime:
            broken = False

            def invoke(self, name, values):
                from cost_model.layouts import buffer_layout
                v = candidates()[0]
                _, a, b, c = inputs_for(v, int(values[buffer_layout(v)['c']] != 0x3f800000))
                m, n, kdim = v['shape']
                output = (C.c_uint32 * 8192)()
                for i in range(m):
                    for j in range(n):
                        x = c[i][j] + sum(a[i][k] * b[k][j] for k in range(kdim))
                        output[i * n + j] = struct.unpack('<I', struct.pack('<f', x))[0]
                if self.broken: output[m * n - 1] = 0
                return output, {}

        runtime = Runtime()
        assert correctness(runtime, candidates()[0])['max_absolute_error'] == 0
        runtime.broken = True
        with pytest.raises(ValueError, match='output'):
            correctness(runtime, candidates()[0])


class TestRanking:

    def test_ties_have_average_rank_and_half_pair_credit(self):
        assert ranks([4, 4, 8]) == [1.5, 1.5, 3]
        result = metrics(['a', 'b', 'c'], [4, 4, 8], dict(a=10, b=20, c=30))
        assert result['pairwise_ranking_accuracy'] == 2.5 / 3
        result = metrics(['a', 'b', 'c'], [16, 16, 16], dict(a=30, b=10, c=20))
        assert result['pairwise_ranking_accuracy'] == .5
        assert result['spearman_rank'] is None
        assert result['top1_tie_regret_range_percent'] == [0, 200]

    def test_top1_regret_and_topk(self):
        result = metrics(['a', 'b', 'c'], [2, 1, 3], dict(a=10, b=20, c=30))
        assert result['top1_regret_percent'] == 100
        assert result['exact_k_coverage'] == {'1': 0, '3': 1, '5': 1}

    def test_selection_uses_full_oracle_when_fastest_is_unpredicted(self):
        result = metrics(['slow', 'slower'], [1, 2], dict(fast=1, slow=100, slower=200),
                         oracle_names=['fast', 'slow', 'slower'])
        assert result['pairwise_ranking_accuracy'] == 1
        assert result['top1_regret_percent'] == 9900
        assert result['top1_tie_regret_range_percent'] == [9900, 9900]
        assert result['top_k_coverage'] == {'1': 0, '3': 2 / 3, '5': 2 / 3}
        assert result['exact_k_coverage'] == {'1': 0, '3': 2 / 3, '5': 2 / 3}


class TestCostLoading:

    @pytest.mark.parametrize('change', ['value', 'missing_interval', 'measurement', 'protocol'])
    def test_reject_result_not_bound_to_analysis_and_run(self, save_cost_result, tmp_path, change):
        out, row = save_cost_result(tmp_path)
        if change == 'value':
            write_json(out / 'cost_result.json', {**row, 'value': 99})
            reason = 'analyzed interval'
        elif change == 'missing_interval':
            analysis = json.loads((out / 'analysis.json').read_text())
            analysis.pop('interval_ticks')
            write_json(out / 'analysis.json', analysis)
            reason = 'analyzed interval'
        elif change == 'measurement':
            run = json.loads((out / 'run.json').read_text())
            run['measurement']['measurement_id'] = 'fmul_f32.independent_issue'
            write_json(out / 'run.json', run)
            reason = 'measurement differs'
        else:
            analysis = json.loads((out / 'analysis.json').read_text())
            analysis['protocol']['name'] = 'slope'
            write_json(out / 'analysis.json', analysis)
            reason = 'protocol differs'
        with pytest.raises(ValueError, match=reason):
            costs_from(tmp_path)

    def test_only_qualified_short_measurements(self, save_cost_result):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = save_cost_result(tmp)
            assert costs_from(tmp)['fadd_f32.independent_issue']['value'] == 4.0
            write_json(out / 'qualification.json', {'passed': False})
            with pytest.raises(ValueError):
                costs_from(tmp)

    def test_reject_slope_and_sequence(self, save_cost_result):
        with tempfile.TemporaryDirectory() as tmp:
            out, row = save_cost_result(tmp)
            for change in ({'timing_protocol':
                            'slope'}, {'metric':
                                       'sequence_cost'}, {'metric': 'protocol_cost', 'unit':
                                                          'ticks/protocol'}, {'native_opcode': ''}):
                write_json(out / 'cost_result.json', {**row, **change})
                with pytest.raises(ValueError, match='invalid primitive CostResult'):
                    costs_from(tmp)
            write_json(out / 'cost_result.json', row)
            write_json(out / 'analysis.json', {'passed': False})
            with pytest.raises(ValueError, match='gate failed'):
                costs_from(tmp)

    def test_no_fallback_input(self, save_cost_result):
        with tempfile.TemporaryDirectory() as tmp:
            out, row = save_cost_result(tmp)
            (out / 'cost_result.json').unlink()
            write_json(out / 'sequence_result.json', row)
            write_json(out / 'provisional_lut.json', {'rates': {'shared_load': {'I': 4, 'D': 64}}})
            write_json(out / 'observation.json', {'interval_ticks': 56, 'scope': 'STS LDS wait feedback'})
            assert costs_from(tmp) == {}


class TestNativeEvidence:

    @pytest.mark.parametrize('llvm_count', [0, 1, 2])
    def test_offline_check_enforces_primitive_counts(self, tmp_path, llvm_count):
        from cost_model.execute import offline_check

        rows = [
            {'op': 'get_time', 'args': ['s0']}, {'op': 'arrive', 'args': ['slcnt(0)']},
            {'op': 'get_time', 'args': ['s2']}, {'op': 'arrive', 'args': ['slcnt(0)']},
            {'op': 'lds_u16', 'args': ['r1', 'r2', '0x0']}, {'op': 'arrive', 'args': ['gvmcnt(0)']},
            {'op': 'get_time', 'args': ['s4']}, {'op': 'arrive', 'args': ['slcnt(0)']},
            {'op': 'endk', 'args': []}
        ]
        for name in ('kernel.s', 'kernel.llir', 'kernel.mlir', 'benchmark.fatbin'):
            (tmp_path / name).write_text('')
        write_json(tmp_path / 'compiler_call.json', {'extra_options': ''})
        write_json(tmp_path / 'run.json', {'expected_mma': 0, 'shared': 0})
        events = [{'native': 'lds_u16', 'count': llvm_count}] if llvm_count else []
        tools = dict(mxcc='mxcc', triple='mxc-metax-macahca', cpu='xcore1000')
        with patch('cost_model.execute.command'), \
             patch('cost_model.execute.device_elf', return_value=b'elf'), \
             patch('cost_model.execute.functions', return_value={'layout_gemm': (0, b'\0' * (8 * len(rows)))}), \
             patch('cost_model.execute.assembly_records', return_value=rows), \
             patch('cost_model.llvm.selected_events', return_value=(events, {})):
            if llvm_count == 1:
                assert offline_check(tmp_path, tools, 'layout_gemm')['passed']
            else:
                with pytest.raises(ValueError, match='LLVM/ISA primitive counts differ: lds_u16'):
                    offline_check(tmp_path, tools, 'layout_gemm')
        evidence = json.loads((tmp_path / 'offline.json').read_text())
        assert evidence['passed'] == (llvm_count == 1)
        assert evidence['llvm_to_isa']['lds_u16']['llvm_count'] == llvm_count
        assert evidence['llvm_to_isa']['lds_u16']['isa_count'] == 1

    def test_fmac_reads_implicit_accumulator(self):
        rows = [{'op': 'fmac_f32', 'args': ['r0', 'r1', 'r2']}, {'op': 'fmac_f32', 'args': ['r0', 'r3', 'r4']}]
        g = isa_graph(rows)
        assert 'r0:lo' in g[0]['reads']
        assert any((e['from'] == 0 and e['component'] == 'r0:lo' for e in g[1]['edges']))
        assert reg_bits('r3_hi') == ['r3:hi']

    def test_both_wait_clauses_and_sts_async_queues(self):
        rows = [{'op': 'sts_b32', 'args':
                 ['r0', '0x0', 'r1']}, {'op': 'ldg_b32', 'args': ['r2', 'r8', '0x0 ret0_en bsm pred_reg:s4']},
                {'op': 'arrive', 'args': ['gvmcnt(0) & bsmcnt(0)']}]
        g = isa_graph(rows)
        assert g[1]['writes'] == []
        assert {(e['from'], e['queue']) for e in g[2]['edges']} == {(0, 'bsmcnt'), (1, 'bsmcnt'), (1, 'gvmcnt')}
        assert pending_queues('sts_b128') == ('bsmcnt', )
        graph = isa_graph([{'op': 'smov_b32', 'args': ['md', 'r1']}, rows[1]])
        assert {'from': 0, 'kind': 'register_RAW', 'component': 'md'} in graph[1]['edges']
        assert wait_clauses('gvmcnt(0) & bsmcnt(2)') == [('gvmcnt', 0), ('bsmcnt', 2)]
        with pytest.raises(ValueError):
            wait_clauses('bsmcnt(0) & bsmcnt(0)')

    def test_counter_completion(self):
        rows = []
        for _ in range(3):
            rows.extend([{'op': 'get_time', 'args': ['s0']}, {'op': 'arrive', 'args': ['slcnt(0)']}])
        rows.insert(4, {'op': 'arrive', 'args': ['gvmcnt(0)']})
        assert counter_window(rows) == [0, 2, 5]
        with pytest.raises(ValueError):
            counter_window(rows[:-1])


class TestLLVMExtraction:

    def test_ssa_edges_and_single_block_boundary(self, llvm_kernel):
        text = llvm_kernel('  %x = fadd float %a, %b\n  %y = fadd float %x, %b\n')
        nodes, start, stop = llvm_nodes(text)
        assert (start, stop) == (1, 4)
        assert nodes[3]['predecessors'] == [2]
        with pytest.raises(ValueError, match='single basic block'):
            llvm_nodes(llvm_kernel('  br label %next\nnext:\n'))

    def test_mma_group_keeps_all_external_a_b_c_dependencies(self, llvm_kernel, mma_call):
        body = '  %x = fadd float %input0, %input1\n  %bits = bitcast float %x to i32\n'
        body += '  %halves = bitcast i32 %bits to <2 x half>\n'
        body += '  %h = extractelement <2 x half> %halves, i32 0\n'
        body += '  %a = insertelement <4 x half> undef, half %h, i32 0\n'
        body += '  %b = insertelement <4 x half> undef, half %h, i32 0\n'
        body += '  %c = insertelement <4 x float> undef, float %x, i32 0\n'
        body += mma_call('first') + mma_call('second', 'first')
        mlir = ('%start = llvm.inline_asm\n'
                '%first = llvm.call @llvm.mxc.mma.f32.16x16x16f16() loc(#loc1)\n'
                '%second = llvm.call @llvm.mxc.mma.f32.16x16x16f16() loc(#loc1)\n'
                '%stop = llvm.inline_asm\n#loc1 = loc("kernel.py":1:1)')
        events, _ = extract_events(llvm_kernel(body), '', mlir)
        assert len(events) == 3 and events[1]['count'] == events[2]['count'] == 1
        assert events[1]['deps'] == [events[0]['id']]
        assert events[2]['deps'] == [events[0]['id'], events[1]['id']]

    def test_unknown_shared_footprint_is_rejected(self, llvm_kernel):
        text = llvm_kernel('  %x = load i32, ptr addrspace(3) %unknown, align 4\n')
        with pytest.raises(ValueError, match='shared alias footprint'):
            extract_events(text, '', '%start = llvm.inline_asm\n%stop = llvm.inline_asm')

    def test_shared_exchange_requires_matching_footprint_and_annotation(self, llvm_kernel):
        body = ('  %lane = call i32 @llvm.mxc.thread.id.x()\n'
                '  %offset = mul i32 %lane, 16\n'
                '  %p = getelementptr i8, ptr addrspace(3) @global_smem, i32 %offset\n'
                '  store <4 x i32> %v, ptr addrspace(3) %p, align 16\n'
                '  %r = load <4 x i32>, ptr addrspace(3) %p, align 16\n')
        mlir = ('%start = llvm.inline_asm\n'
                'llvm.store %v, %p : !llvm.ptr<3> loc(#loc1)\n'
                '%r = llvm.load %p : !llvm.ptr<3> loc(#loc2)\n'
                '%stop = llvm.inline_asm\n'
                '#loc1 = loc("kernel.py":1:1)\n#loc2 = loc("kernel.py":1:1)')
        conversion = ('%r = ttg.convert_layout %x : tensor<16x16xi32, #a> -> tensor<16x16xi32, #b> loc(#loc1)\n'
                      '#loc1 = loc("kernel.py":1:1)')
        events, _ = extract_events(llvm_kernel(body), conversion, mlir)
        assert len(events) == 1 and events[0]['kind'] == 'rearrange'
        assert events[0]['stores'] == events[0]['loads'] == 1
        assert events[0]['deps'] == []
        # A larger conversion can reuse one scratch range in several rounds.
        repeated_body = body + ('  store <4 x i32> %v, ptr addrspace(3) %p, align 16\n'
                                '  %r2 = load <4 x i32>, ptr addrspace(3) %p, align 16\n')
        repeated_mlir = mlir.replace(
            '%stop = llvm.inline_asm', 'llvm.store %v, %p : !llvm.ptr<3> loc(#loc1)\n'
            '%r2 = llvm.load %p : !llvm.ptr<3> loc(#loc2)\n%stop = llvm.inline_asm')
        larger = conversion.replace('16x16', '16x32')
        rounds, _ = extract_events(llvm_kernel(repeated_body), larger, repeated_mlir)
        assert len(rounds) == 2
        assert sum(e['exchange_bytes'] for e in rounds) == 16 * 32 * 4
        with pytest.raises(ValueError, match='layout conversions'):
            extract_events(llvm_kernel(repeated_body), conversion, repeated_mlir)
        with pytest.raises(ValueError, match='layout conversions'):
            extract_events(llvm_kernel(body), '', mlir)

    def test_nested_shared_gep_and_unknown_do_not_alias_by_default(self, make_nodes):
        n = make_nodes(
            '%lane = call i32 @llvm.mxc.thread.id.x()', '%byte = shl i32 %lane, 1',
            '%p = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 4096), i32 %byte',
            '%q = getelementptr i8, ptr addrspace(3) @global_smem, i32 %unknown')
        _, pointers = address_values(n)
        assert pointers['%p'] == ('shared', [4096 + lane * 2 for lane in range(64)])
        assert '%q' not in pointers

    def test_f16_predicator_width_and_actual_nonvolatile_form(self, make_nodes):
        n = make_nodes(
            'call void @dummy()',
            '%x = call half @llvm.mxc.ldg.predicator.f16(ptr addrspace(1) %p, i32 0, i64 %mask, i1 true, i1 true, i1 false, i1 false)',
            '%s = load <4 x i16>, ptr addrspace(3) %q, align 8', 'ret void')
        m = memory_accesses(n, 0, 3,
                            {'%p': ('input', list(range(64))), '%q': ('shared', [lane * 8 for lane in range(64)])})
        assert [row['access_bytes'] for row in m] == [2, 8]
        assert not m[1]['volatile']
        wide = make_nodes(
            'call void @dummy()',
            '%v = call <4 x i32> @llvm.mxc.ldg.predicator.v4i32(ptr addrspace(1) %p, i32 0, i64 %mask, i1 true, i1 true, i1 false, i1 false)',
            'ret void')
        assert memory_accesses(wide, 0, 2, {'%p': ('input', list(range(64)))})[0]['access_bytes'] == 16

    def test_native_address_extraction_and_bitwise_fusion(self):
        values = {'r0': tuple(range(64))}
        active = [True] * 64
        numeric_step({'op': 'bfe_i32', 'args': ['r1', 'r0', '4', '1']}, values, active)
        assert values['r1'] == tuple((4294967295 if lane & 16 else 0 for lane in range(64)))
        numeric_step({'op': 'or3_b32', 'args': ['r2', 'r0', '0x80', '0x100']}, values, active)
        assert values['r2'] == tuple((lane | 384 for lane in range(64)))

    def test_unknown_native_write_does_not_preserve_stale_raw(self):
        graph = isa_graph([{'op': 'mov_b32', 'args': ['r1', 'r0']}, {'op': 'unsupported', 'args': ['r1', 'r2']},
                           {'op': 'add_u32', 'args': ['r3', 'r1', 'r4']}])
        assert not graph[1]['register_analysis_supported']
        assert graph[2]['edges'] == []

    def test_fixed_layout_inputs(self):
        rows = candidates()
        assert len(rows) == 24
        for v in rows:
            _, a, b, c = inputs_for(v, 1)
            m, n, k = v['shape']
            assert (len(a), len(a[0]), len(b), len(b[0]), len(c)) == (m, k, k, n, m)

    def test_memory_form_keeps_vector_volatile(self):
        row = memory_form('store volatile <1 x i32> %v, ptr addrspace(3) %p, align 4')
        assert row['form'] == '<1 x i32>'
        assert row['volatile']


class TestTimeline:

    def test_missing_primitive_cost_does_not_fall_back(self, issue_event):
        with pytest.raises(ValueError, match='primitive cost missing'):
            predict([issue_event(0)], {}, MEMORY_D)

    def test_tile_issue_and_last_instruction_tail(self, issue_event):
        sim = Timeline()
        sim.issue(issue_event(0, count=3), 4, 20)
        assert sim.clock == 12
        assert sim.pending == {0: 16}

    def test_independent_work_then_raw(self, issue_event):
        sim = Timeline()
        sim.issue(issue_event(0), 4, 20)
        sim.issue(issue_event(1), 4, 4)
        sim.issue(issue_event(2, deps=(0, )), 4, 4)
        assert sim.clock == 24
        assert sim.trace[-1]['raw_wait'] == 12

    def test_sync_drains_max_outstanding_without_wait_cpi(self, issue_event):
        sim = Timeline()
        sim.issue(issue_event(0), 4, 32)
        sim.issue(issue_event(1), 4, 16)
        assert sim.drain() == 24
        assert sim.clock == 32
        assert not sim.pending

    def test_rearrange_drains_then_completes_stores_and_issues_loads(self, issue_event):
        events = [
            issue_event(0), {
                'id': 1, 'kind': 'rearrange', 'stores': 2, 'loads': 3, 'store_native': 'sts_b128', 'load_native':
                'lds_b128'
            },
            issue_event(2, deps=(1, ))
        ]
        lut = {
            'add_f32': {'I': 4, 'D': 20, 'source': 'test'}, 'shared_store': {'I': 8, 'D': None, 'source': 'test'},
            'shared_load': {'I': 4, 'D': None, 'source': 'test'}
        }
        result = predict(events, lut, {'sts_d': 32, 'lds_d': 64, 'gvm_d': 256})
        assert result['predicted_cost'] == 136
        assert result['trace'][1]['drain'] == 16

    def test_missing_nonmemory_d_is_not_silently_set_to_i(self, issue_event):
        with pytest.raises(ValueError, match='unmeasured non-memory D'):
            predict([issue_event(0)], {'add_f32': {'I': 4, 'D': None, 'source': 'test'}}, {})

    def test_candidates_do_not_share_pending_state(self, issue_event):
        a = Timeline()
        a.issue(issue_event(0), 4, 256)
        b = Timeline()
        assert b.drain() == 0

    def test_view_and_address_work_is_zero_but_forwards_raw(self, llvm_kernel):
        llvm = llvm_kernel(
            ' %x = fadd float %a, %b\n %bits = bitcast float %x to i32\n %address = add i32 %bits, 4\n %view = bitcast i32 %address to float\n fence seq_cst\n %y = fadd float %view, %b\n'
        )
        mlir = '%start = llvm.inline_asm\n%stop = llvm.inline_asm'
        with pytest.raises(ValueError, match='unpriced integer data operation'):
            extract_events(llvm, '', mlir)
        llvm = llvm.replace(' %address = add i32 %bits, 4\n', '').replace('bitcast i32 %address to float',
                                                                          'bitcast i32 %bits to float')
        events, ignored = extract_events(llvm, '', mlir)
        assert len(events) == 2
        assert events[1]['deps'] == [events[0]['id']]
        assert ignored['fence'] == 1

    def test_unannotated_shuffle_is_not_assumed_free(self, llvm_kernel):
        llvm = llvm_kernel(' %x = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> zeroinitializer\n')
        with pytest.raises(ValueError, match='unmapped shufflevector'):
            extract_events(llvm, '', '%start = llvm.inline_asm\n%stop = llvm.inline_asm')

    def test_unsupported_work_is_not_zero_cost_scaffolding(self, llvm_kernel, subtests):
        for operation in ('%x = fdiv float %a, %b', '%x = call float @unknown(float %a)',
                          'call void asm sideeffect "snop 0", ""()'):
            llvm = llvm_kernel('  ' + operation + '\n')
            with subtests.test(operation=operation), pytest.raises(ValueError, match='unsupported LLVM operation'):
                extract_events(llvm, '', '%start = llvm.inline_asm\n%stop = llvm.inline_asm')


class TestCatalog:

    def test_minimal_records(self, catalog):
        for row in catalog['mappings']:
            assert set(row) == FIELDS
        assert len({r['mapping_id'] for r in catalog['mappings']}) == len(catalog['mappings'])

    def test_queries_return_branches_not_a_selected_lowering(self, catalog):
        rows = query_catalog(catalog, source='tt.fp_to_fp')
        assert len(rows) > 1
        assert query_catalog(catalog, llvm='llvm.mxc.rcp.f32')
        assert query_catalog(catalog, source='does_not_exist') == []
        assert 'mapping_id' in table(rows)
        assert query_catalog(catalog, text='RTNE')

    def test_source_names_do_not_match_unrelated_api_prefixes(self, catalog):
        rows = query_catalog(catalog, source='tl.dot')
        assert len(rows) == 11
        assert not any(('tl.dot_scaled' in r['source']['apis'] for r in rows))
        assert query_catalog(catalog, source='tl.dot_scaled')
        assert query_catalog(catalog, source='tl.do') == []

    def test_load_query_shows_rules_instead_of_width_products(self, catalog):
        rows = query_catalog(catalog, source='tl.load')
        assert {r['mapping_id']
                for r in rows} == {
                    'global.load.predicator', 'global.load.fallback.ordinary', 'global.load.fallback.volatile',
                    'shared.load.unmasked', 'shared.load.predicated'
                }
        shared = {r['mapping_id']: r for r in catalog['mappings'] if r['mapping_id'].startswith('shared.')}
        assert 'V=1: y:T' in shared['shared.load.unmasked']['llvm']
        assert '<1 x T>' in shared['shared.store']['llvm']
        assert 'else: zero' in shared['shared.load.predicated']['llvm']
        assert 'alignment=V*bits(T)/8' in str(shared['shared.load.unmasked']['conditions'])

    def test_bf16_cast_direction_and_rounding_remain_distinct(self, catalog):
        narrow = query_catalog(catalog, source='tensor.to', text='f32 -> bf16')
        assert {r['mapping_id'] for r in narrow} == {'bf16.narrow.rtz', 'bf16.narrow.rtne', 'bf16.narrow.rtne_no_nan'}
        by_id = {r['mapping_id']: r for r in narrow}
        assert 'lshr' in by_id['bf16.narrow.rtz']['llvm']
        assert 'llvm.mxc.cvt.f32tobf16' in by_id['bf16.narrow.rtne']['llvm']
        assert '32767' in by_id['bf16.narrow.rtne_no_nan']['llvm']
        assert [r['mapping_id']
                for r in query_catalog(catalog, source='tensor.to', text='bf16 -> f32')] == ['bf16.widen']

    def test_packed_bindings_and_communication_packets_are_not_merged(self, catalog):
        by_id = {r['mapping_id']: r for r in catalog['mappings']}
        assert 'B=<float 1,float 1>' in by_id['packed.add']['llvm']
        assert 'C=zeroinitializer' in by_id['packed.mul']['llvm']
        assert 'C=accumulator' in by_id['packed.fma']['llvm']
        assert len([i for i in by_id if i.startswith('mma.')]) == 9
        assert len([i for i in by_id if i.startswith('fp8.')]) == 16
        for mid in ('shared.load.perm.integrated', 'shared.bsm_perm.explicit'):
            assert '0x01000504' in by_id[mid]['llvm']
            assert '0x03020706' in by_id[mid]['llvm']
        assert 'llvm.mxc.byte.perm' not in by_id['shared.load.perm.split']['llvm']

    def test_parameterized_arithmetic_keeps_specialized_branches(self, catalog):
        by_id = {r['mapping_id']: r for r in catalog['mappings']}
        assert 'T in {half, float, double}' in str(by_id['float.fadd']['conditions'])
        assert 'TRITON_DISABLE_ELEMENTWISE_PK_FMA_OPT' in str(by_id['float.fadd']['conditions'])
        assert 'BF16 bridge is a separate rule' in str(by_id['float.fadd']['conditions'])
        assert '(double,half)' in str(by_id['cast.fptrunc']['conditions'])
        assert 'llvm.mxc.rcp.f16' in by_id['math.rcp']['llvm']
        assert 'llvm.mxc.rcp.f32' in by_id['math.rcp']['llvm']
        for mid, opcode in [('atomic.atomic_add.integer', 'add'), ('atomic.atomic_add.floating', 'fadd'),
                            ('atomic.atomic_max.signed', 'max'), ('atomic.atomic_max.unsigned', 'umax')]:
            assert 'atomicrmw ' + opcode + ' ' in by_id[mid]['llvm']

    def test_helper_signatures_are_listed_without_expanding_a_library_index(self, catalog):
        rows = query_catalog(catalog, source='tl.extra.cuda.libdevice.sin')
        assert [r['mapping_id'] for r in rows] == ['libdevice.sin']
        assert '@mc_math_func_sinf(fp32)' in rows[0]['llvm']
        assert '@mc_math_func_sin(fp64)' in rows[0]['llvm']
        assert query_catalog(catalog, source='tl.extra.cuda.libdevice.fast_sinf') == rows
        helpers = [r for r in catalog['mappings'] if r['mapping_id'].startswith('libdevice.')]
        assert len(helpers) == 18

    def test_audit_only_descriptions_are_not_runtime_records(self, catalog):
        ids = {r['mapping_id'] for r in catalog['mappings']}
        assert not any((i.startswith(('structural.', 'view.', 'diagnostic.', 'commonir.')) for i in ids))
        assert 'pointer.addptr' in ids
        assert 'layout.block' in ids
        assert 'layout.register' not in ids
        assert 'layout.equivalent' not in ids
        assert 'builtin.loadv4' not in ids
        assert 'builtin.storev4' not in ids

    def test_source_references_in_bound_checkout(self, catalog):
        root = Path(__file__).resolve().parents[4]
        for row in catalog['mappings']:
            for ref in row['source_reference']:
                path = root / ref['path']
                assert path.is_file(), ref
                if ref.get('symbol'):
                    assert ref['symbol'] in path.read_text(errors='replace'), ref

    def test_invalid_data_is_not_silently_ignored(self, catalog, subtests):
        variants = []
        duplicate = copy.deepcopy(catalog)
        duplicate['mappings'].append(duplicate['mappings'][0])
        variants.append(duplicate)
        extra = copy.deepcopy(catalog)
        extra['mappings'][0]['ticks'] = 4
        variants.append(extra)
        missing = copy.deepcopy(catalog)
        del missing['mappings'][0]['llvm']
        variants.append(missing)
        path = copy.deepcopy(catalog)
        path['mappings'][0]['source_reference'][0]['path'] = '../outside'
        variants.append(path)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'catalog.yaml'
            for value in variants:
                with subtests.test(value=list(value)):
                    output.write_text(yaml.safe_dump(value))
                    with pytest.raises(ValueError):
                        load_catalog(output)

    def test_catalog_and_measurement_are_independent(self):
        import ast
        root = Path(__file__).resolve().parents[1]
        for package, forbidden in [('catalog', ('measurement', 'triton')), ('measurement', ('catalog', 'triton'))]:
            for path in (root / package).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        assert node.module.split('.')[0] not in forbidden
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            assert alias.name.split('.')[0] not in forbidden
