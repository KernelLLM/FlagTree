# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def mhc_head_compute_mix_bwd_asc(output_mix_grad: T.handle, input_mix: T.handle, mhc_scale: T.handle, mhc_base: T.handle, input_mix_grad: T.handle, mhc_scale_grad_partial: T.handle, mhc_base_grad_partial: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "launch_grid_only": True, "max_vector_bits": 64, "supports_kernel_status_return": False, "supports_vector_predicate": True, "tag": ""})})
        num_tokens = T.int32()
        output_mix_grad_1 = T.match_buffer(output_mix_grad, (num_tokens, 4), strides=(4, 1))
        input_mix_1 = T.match_buffer(input_mix, (num_tokens, 4), strides=(4, 1))
        mhc_scale_1 = T.match_buffer(mhc_scale, (1,), strides=(1,))
        mhc_base_1 = T.match_buffer(mhc_base, (4,), strides=(1,))
        input_mix_grad_1 = T.match_buffer(input_mix_grad, (num_tokens, 4), strides=(4, 1))
        mhc_scale_grad_partial_1 = T.match_buffer(mhc_scale_grad_partial, (20, 1), strides=(1, 1))
        mhc_base_grad_partial_1 = T.match_buffer(mhc_base_grad_partial, (20, 4), strides=(4, 1))
        with T.sblock("root"):
            T.reads()
            T.writes()
            mhc_scale_grad_frag = T.Buffer((1,), scope="local.fragment")
            input_recompute_frag = T.Buffer((32, 4), scope="local.fragment")
            grad_frag = T.Buffer((32, 4), scope="local.fragment")
            mhc_base_grad_frag = T.Buffer((4,), scope="local.fragment")
            T.sblock_attr({"layout_map": {mhc_scale_grad_frag: metadata["tl.Fragment"][0], input_recompute_frag: metadata["tl.Fragment"][1], grad_frag: metadata["tl.Fragment"][2], mhc_base_grad_frag: metadata["tl.Fragment"][3]}})
            T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `output_mix_grad`, `input_mix`, `input_mix_grad`")
            bx = T.launch_thread("blockIdx.x", 20)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                T.sblock_attr({"layout_map": {mhc_scale_grad_frag: metadata["tl.Fragment"][0], input_recompute_frag: metadata["tl.Fragment"][1], grad_frag: metadata["tl.Fragment"][2], mhc_base_grad_frag: metadata["tl.Fragment"][3]}})
                T.attr({"stage": 0}, "tl.schedule_unit", 1)
                T.attr({"core_mask": T.int64(1), "ii": T.int64(2689), "latency": T.int64(2689)}, "tl.ascend_task", 1)
                with T.sblock("SIMT_VF", no_realize=True):
                    T.reads()
                    T.writes()
                    mhc_scale_grad_reducer = T.Buffer((1,), scope="local.fragment")
                    mhc_base_grad_reducer = T.Buffer((4,), scope="local.fragment")
                    T.sblock_attr({"layout_map": {mhc_scale_grad_frag: metadata["tl.Fragment"][0], input_recompute_frag: metadata["tl.Fragment"][1], grad_frag: metadata["tl.Fragment"][2], mhc_base_grad_frag: metadata["tl.Fragment"][3], mhc_scale_grad_reducer: metadata["tl.Fragment"][4], mhc_base_grad_reducer: metadata["tl.Fragment"][5]}, "tl.vf_source_index": T.int64(0)})
                    mhc_scale_grad_reducer = T.sblock_alloc_buffer((1,), data=mhc_scale_grad_reducer.data, scope="local.fragment")
                    mhc_base_grad_reducer = T.sblock_alloc_buffer((4,), data=mhc_base_grad_reducer.data, scope="local.fragment")
                    mhc_scale_grad_frag = T.sblock_alloc_buffer((1,), data=mhc_scale_grad_frag.data, scope="local.fragment")
                    mhc_base_grad_frag = T.sblock_alloc_buffer((4,), data=mhc_base_grad_frag.data, scope="local.fragment")
                    grad_frag = T.sblock_alloc_buffer((32, 4), data=grad_frag.data, scope="local.fragment")
                    input_recompute_frag = T.sblock_alloc_buffer((32, 4), data=input_recompute_frag.data, scope="local.fragment")
                    simtvf_tx = T.launch_thread("threadIdx.x", 128)
                    simtvf_ty = T.launch_thread("threadIdx.y", 1)
                    simtvf_tz = T.launch_thread("threadIdx.z", 1)
                    T.attr("simtvf", "tl.simtvf_scope", 1)
                    T.fill(T.region(mhc_scale_grad_reducer[0], 2, 1), T.float32(0.0))
                    T.fill(T.region(mhc_base_grad_reducer[0], 2, 4), T.float32(0.0))
                    for w in range((num_tokens + 639) // 640):
                        if w * 20 + bx < (num_tokens + 31) // 32:
                            for i1 in T.parallel(32, annotations={"parallel_loop_layout": metadata["tl.Fragment"][6]}):
                                for j in T.parallel(4):
                                    if w * 640 + bx * 32 + i1 < num_tokens:
                                        input_recompute_frag[i1, j] = T.sigmoid(input_mix_1[w * 640 + bx * 32 + i1, j] * mhc_scale_1[0] + mhc_base_1[j])
                                        grad_frag[i1, j] = input_recompute_frag[i1, j] * (T.float32(1.0) - input_recompute_frag[i1, j]) * output_mix_grad_1[w * 640 + bx * 32 + i1, j]
                                        input_mix_grad_1[w * 640 + bx * 32 + i1, j] = grad_frag[i1, j] * mhc_scale_1[0]
                                        mhc_scale_grad_reducer[0] = mhc_scale_grad_reducer[0] + grad_frag[i1, j] * input_mix_1[w * 640 + bx * 32 + i1, j]
                                        mhc_base_grad_reducer[j] = mhc_base_grad_reducer[j] + grad_frag[i1, j]
                    T.finalize_reducer(T.region(mhc_scale_grad_reducer[0], 3, 1), 0, 128, 4, 4, 1, plan=1)
                    T.copy(T.region(mhc_scale_grad_reducer[0], 1, 1), T.region(mhc_scale_grad_frag[0], 2, 1))
                    T.finalize_reducer(T.region(mhc_base_grad_reducer[0], 3, 4), 0, 128, 4, plan=1)
                    T.copy(T.region(mhc_base_grad_reducer[0], 1, 4), T.region(mhc_base_grad_frag[0], 2, 4))
                    T.ascend_copy(T.region(mhc_scale_grad_frag[0], 1, 1), T.region(mhc_scale_grad_partial_1[bx, 0], 2, 1, 1))
                    T.ascend_copy(T.region(mhc_base_grad_frag[0], 1, 4), T.region(mhc_base_grad_partial_1[bx, 0], 2, 1, 4))

# Metadata omitted. Use show_meta=True in script() method to show it.