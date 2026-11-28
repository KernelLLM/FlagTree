# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def mhc_head_compute_mix_bwd_asc(output_mix_grad: T.handle, input_mix: T.handle, mhc_scale: T.handle, mhc_base: T.handle, input_mix_grad: T.handle, mhc_scale_grad_partial: T.handle, mhc_base_grad_partial: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "launch_grid_only": True, "max_vector_bits": 64, "supports_kernel_status_return": False, "supports_vector_predicate": True, "tag": ""}), "tl.has_tma": T.bool(False)})
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
            input_recompute_frag = T.Buffer((1,), scope="local")
            grad_frag = T.Buffer((1,), scope="local")
            mhc_scale_grad_reducer = T.Buffer((1,), scope="local")
            mhc_base_grad_reducer = T.Buffer((1,), scope="local")
            mhc_scale_grad_frag = T.Buffer((1,), scope="local")
            mhc_base_grad_frag = T.Buffer((1,), scope="local")
            T.sblock_attr({"layout_map": {input_recompute_frag: metadata["tl.Fragment"][0], grad_frag: metadata["tl.Fragment"][1], mhc_scale_grad_reducer: metadata["tl.Fragment"][2], mhc_base_grad_reducer: metadata["tl.Fragment"][3], mhc_scale_grad_frag: metadata["tl.Fragment"][4], mhc_base_grad_frag: metadata["tl.Fragment"][5]}})
            T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `output_mix_grad`, `input_mix`, `input_mix_grad`")
            bx = T.launch_thread("blockIdx.x", 20)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                T.sblock_attr({"layout_map": {input_recompute_frag: metadata["tl.Fragment"][0], grad_frag: metadata["tl.Fragment"][1], mhc_scale_grad_reducer: metadata["tl.Fragment"][2], mhc_base_grad_reducer: metadata["tl.Fragment"][3], mhc_scale_grad_frag: metadata["tl.Fragment"][4], mhc_base_grad_frag: metadata["tl.Fragment"][5]}})
                T.attr({}, "tl.buffer_alias_map", 1)
                with T.sblock("VECTOR", no_realize=True):
                    T.reads()
                    T.writes()
                    with T.sblock("SIMT_VF", no_realize=True):
                        T.reads()
                        T.writes()
                        T.sblock_attr({"layout_map": {input_recompute_frag: metadata["tl.Fragment"][0], grad_frag: metadata["tl.Fragment"][1], mhc_scale_grad_reducer: metadata["tl.Fragment"][2], mhc_base_grad_reducer: metadata["tl.Fragment"][3], mhc_scale_grad_frag: metadata["tl.Fragment"][4], mhc_base_grad_frag: metadata["tl.Fragment"][5]}, "tl.vf_source_index": T.int64(0)})
                        mhc_scale_grad_reducer = T.sblock_alloc_buffer((1,), data=mhc_scale_grad_reducer.data, scope="local")
                        mhc_base_grad_reducer = T.sblock_alloc_buffer((1,), data=mhc_base_grad_reducer.data, scope="local")
                        mhc_scale_grad_frag = T.sblock_alloc_buffer((1,), data=mhc_scale_grad_frag.data, scope="local")
                        mhc_base_grad_frag = T.sblock_alloc_buffer((1,), data=mhc_base_grad_frag.data, scope="local")
                        grad_frag = T.sblock_alloc_buffer((1,), data=grad_frag.data, scope="local")
                        input_recompute_frag = T.sblock_alloc_buffer((1,), data=input_recompute_frag.data, scope="local")
                        workspace = T.sblock_alloc_buffer((128,), scope="shared.dyn")
                        workspace_1 = T.sblock_alloc_buffer((128,), scope="shared.dyn")
                        simtvf_tx = T.launch_thread("threadIdx.x", 128)
                        simtvf_ty = T.launch_thread("threadIdx.y", 1)
                        simtvf_tz = T.launch_thread("threadIdx.z", 1)
                        T.attr("simtvf", "tl.simtvf_scope", 1)
                        if simtvf_tx < 128 and simtvf_tx >= 0:
                            for i in T.unroll(1):
                                mhc_scale_grad_reducer[0] = T.float32(0.0)
                        if simtvf_tx < 128 and simtvf_tx >= 0:
                            for i in T.unroll(1):
                                mhc_base_grad_reducer[0] = T.float32(0.0)
                        for w in range((num_tokens + 639) // 640):
                            if w * 20 + bx < (num_tokens + 31) // 32:
                                for i in T.unroll(1):
                                    if w * 640 + bx * 32 + simtvf_tx // 4 < num_tokens:
                                        input_recompute_frag[0] = T.sigmoid(input_mix_1[w * 640 + bx * 32 + simtvf_tx // 4, simtvf_tx % 4] * mhc_scale_1[0] + mhc_base_1[simtvf_tx % 4])
                                        grad_frag[0] = input_recompute_frag[0] * (T.float32(1.0) - input_recompute_frag[0]) * output_mix_grad_1[w * 640 + bx * 32 + simtvf_tx // 4, simtvf_tx % 4]
                                        input_mix_grad_1[w * 640 + bx * 32 + simtvf_tx // 4, simtvf_tx % 4] = grad_frag[0] * mhc_scale_1[0]
                                        mhc_scale_grad_reducer[0] = mhc_scale_grad_reducer[0] + grad_frag[0] * input_mix_1[w * 640 + bx * 32 + simtvf_tx // 4, simtvf_tx % 4]
                                        mhc_base_grad_reducer[0] = mhc_base_grad_reducer[0] + grad_frag[0]
                        for __finred_0 in T.parallel(1):
                            mhc_scale_grad_reducer[__finred_0] = T.call_extern("float32", "tl::AscendAllReduce<tl::SumOp, 128, 4, 0>::run", mhc_scale_grad_reducer[__finred_0], T.tvm_access_ptr(T.type_annotation("float32"), workspace.data, 0, 128, 3))
                        for __finred_0 in T.parallel(1):
                            mhc_scale_grad_reducer[__finred_0] = T.call_extern("float32", "tl::AscendAllReduce<tl::SumOp, 4, 1, 0>::run", mhc_scale_grad_reducer[__finred_0])
                        if simtvf_tx < 128 and simtvf_tx >= 0:
                            for i in T.unroll(1):
                                for i_1 in range(1):
                                    mhc_scale_grad_frag[0] = mhc_scale_grad_reducer[0]
                        for __finred_0 in T.parallel(1):
                            mhc_base_grad_reducer[__finred_0] = T.call_extern("float32", "tl::AscendAllReduce<tl::SumOp, 128, 4, 0>::run", mhc_base_grad_reducer[__finred_0], T.tvm_access_ptr(T.type_annotation("float32"), workspace_1.data, 0, 128, 3))
                        if simtvf_tx < 128 and simtvf_tx >= 0:
                            for i in T.unroll(1):
                                mhc_base_grad_frag[0] = mhc_base_grad_reducer[0]
                        if simtvf_tx == 0 and (simtvf_tx < 128 and simtvf_tx >= 0):
                            for i in T.unroll(1):
                                for i_1 in range(1):
                                    mhc_scale_grad_partial_1[bx, 0] = mhc_scale_grad_frag[0]
                        if simtvf_tx // 4 == 0 and (simtvf_tx < 128 and simtvf_tx >= 0):
                            for i in T.unroll(1):
                                mhc_base_grad_partial_1[bx, simtvf_tx % 4] = mhc_base_grad_frag[0]

# Metadata omitted. Use show_meta=True in script() method to show it.