# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def mhc_head_compute_mix_fwd_asc(input_mix: T.handle, mhc_scale: T.handle, mhc_base: T.handle, output_mix: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "launch_grid_only": True, "max_vector_bits": 64, "supports_kernel_status_return": False, "supports_vector_predicate": True, "tag": ""})})
        num_tokens = T.int32()
        input_mix_1 = T.match_buffer(input_mix, (num_tokens, 4), strides=(4, 1))
        mhc_scale_1 = T.match_buffer(mhc_scale, (1,), strides=(1,))
        mhc_base_1 = T.match_buffer(mhc_base, (4,), strides=(1,))
        output_mix_1 = T.match_buffer(output_mix, (num_tokens, 4), strides=(4, 1))
        with T.sblock("root"):
            T.reads()
            T.writes()
            input_frag = T.Buffer((32, 4), scope="local.fragment")
            T.sblock_attr({"layout_map": {input_frag: metadata["tl.Fragment"][0]}})
            T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `input_mix`, `output_mix`")
            bx = T.launch_thread("blockIdx.x", 20)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                input_ub = T.handle("float32", "shared.dyn")
                T.sblock_attr({"layout_map": {input_frag: metadata["tl.Fragment"][0]}, "tl.buffer_versions_map": {input_ub: 2}})
                input_ub_1 = T.sblock_alloc_buffer((32, 4), data=input_ub, scope="shared.dyn")
                input_ub_version_counter_1 = T.sblock_alloc_buffer((1,), "int32", scope="local.var")
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(1)}, "tl.ascend_task", 1)
                    input_ub_version_counter_1[0] = 0
                T.attr({"stage": 0}, "tl.schedule_unit", 1)
                for w in T.serial(((((num_tokens + 31) // 32 - 1) // T.max(1, T.min(8, (num_tokens + 31) // 32)) + 1) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + 19) // 20, annotations={"multi_buffer_eligible": [input_ub], "num_stages": 2, "tl.multi_buffer_counter_map": {input_ub: input_ub_version_counter_1}, "tl.storage_epoch_guard_map": {input_ub: __cond_0, output_mix_1.data: __cond_0}}):
                    __cond_0 = T.bool()
                    with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                        T.attr({"core_mask": T.int64(1), "ii": T.int64(1), "latency": T.int64(1)}, "tl.ascend_task", 1)
                        __cond_0 = T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32
                    with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                        if __cond_0:
                            T.attr({"core_mask": T.int64(1), "ii": T.int64(13), "latency": T.int64(101)}, "tl.ascend_task", 1)
                            T.ascend_copy(T.region(input_mix_1[T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32, 0], 1, T.min(32, T.max(num_tokens - T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 - T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32, 0)), 4), T.region(input_ub_1[0, 0], 2, T.min(32, T.max(num_tokens - T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 - T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32, 0)), 4))
                    with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                        if __cond_0:
                            T.attr({"core_mask": T.int64(1), "ii": T.int64(59), "latency": T.int64(59)}, "tl.ascend_task", 1)
                            with T.sblock("SIMT_VF", no_realize=True):
                                T.reads()
                                T.writes()
                                T.sblock_attr({"layout_map": {input_frag: metadata["tl.Fragment"][0]}, "tl.vf_source_index": T.int64(0)})
                                input_frag = T.sblock_alloc_buffer((32, 4), data=input_frag.data, scope="local.fragment")
                                simtvf_tx = T.launch_thread("threadIdx.x", 256)
                                simtvf_ty = T.launch_thread("threadIdx.y", 1)
                                simtvf_tz = T.launch_thread("threadIdx.z", 1)
                                T.attr("simtvf", "tl.simtvf_scope", 1)
                                T.ascend_copy(T.region(input_ub_1[0, 0], 1, 32, 4), T.region(input_frag[0, 0], 2, 32, 4))
                                for i in T.parallel(32, annotations={"parallel_loop_layout": metadata["tl.Fragment"][1], "parallel_loop_predicate": simtvf_tx // 128 == 0}):
                                    for j in T.parallel(4):
                                        if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32 + i < num_tokens:
                                            output_mix_1[T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32 + i, j] = T.sigmoid(input_frag[i, j] * mhc_scale_1[0] + mhc_base_1[j]) + T.float32(9.9999999999999995e-07)
                    T.attr({"stage": 0}, "tl.schedule_unit", 1)
                    T.attr({"core_mask": T.int64(1)}, "tl.ascend_task", 1)
                    if __cond_0:
                        input_ub_version_counter_1[0] = input_ub_version_counter_1[0] + 1

# Metadata omitted. Use show_meta=True in script() method to show it.