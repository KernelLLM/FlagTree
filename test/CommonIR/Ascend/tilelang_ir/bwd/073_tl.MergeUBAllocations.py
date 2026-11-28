# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def mhc_head_compute_mix_bwd_asc(output_mix_grad: T.handle, input_mix: T.handle, mhc_scale: T.handle, mhc_base: T.handle, input_mix_grad: T.handle, mhc_scale_grad_partial: T.handle, mhc_base_grad_partial: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "launch_grid_only": True, "max_vector_bits": 64, "supports_kernel_status_return": False, "supports_vector_predicate": True, "tag": ""}), "tirx.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1, 2, 3, 4, 5, 6]})
        num_tokens = T.int32()
        output_mix_grad_1 = T.match_buffer(output_mix_grad, (num_tokens, 4), strides=(4, 1))
        input_mix_1 = T.match_buffer(input_mix, (num_tokens, 4), strides=(4, 1))
        mhc_scale_1 = T.match_buffer(mhc_scale, (1,), strides=(1,))
        mhc_base_1 = T.match_buffer(mhc_base, (4,), strides=(1,))
        input_mix_grad_1 = T.match_buffer(input_mix_grad, (num_tokens, 4), strides=(4, 1))
        mhc_scale_grad_partial_1 = T.match_buffer(mhc_scale_grad_partial, (20, 1), strides=(1, 1))
        mhc_base_grad_partial_1 = T.match_buffer(mhc_base_grad_partial, (20, 4), strides=(4, 1))
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `output_mix_grad`, `input_mix`, `input_mix_grad`")
        Module.mhc_head_compute_mix_bwd_asc_kernel(input_mix_1.data, input_mix_grad_1.data, mhc_base_1.data, mhc_base_grad_partial_1.data, mhc_scale_1.data, mhc_scale_grad_partial_1.data, output_mix_grad_1.data, num_tokens)

    @T.prim_func(private=True)
    def mhc_head_compute_mix_bwd_asc_kernel(input_mix: T.handle("float32", "global"), input_mix_grad: T.handle("float32", "global"), mhc_base: T.handle("float32", "global"), mhc_base_grad_partial: T.handle("float32", "global"), mhc_scale: T.handle("float32", "global"), mhc_scale_grad_partial: T.handle("float32", "global"), output_mix_grad: T.handle("float32", "global"), num_tokens: T.int32):
        T.func_attr({"target": T.target({"keys": ["ascend"], "kind": "ascend", "launch_grid_only": True, "max_vector_bits": 64, "supports_kernel_status_return": False, "supports_vector_predicate": True, "tag": ""}), "tirx.is_global_func": True, "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 2, 4, 6]})
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `output_mix_grad`, `input_mix`, `input_mix_grad`")
        mhc_base_grad_partial_1 = T.decl_buffer((80,), data=mhc_base_grad_partial)
        mhc_scale_grad_partial_1 = T.decl_buffer((20,), data=mhc_scale_grad_partial)
        input_mix_grad_1 = T.decl_buffer((num_tokens * 4,), data=input_mix_grad)
        output_mix_grad_1 = T.decl_buffer((num_tokens * 4,), data=output_mix_grad)
        mhc_base_1 = T.decl_buffer((4,), data=mhc_base)
        mhc_scale_1 = T.decl_buffer((1,), data=mhc_scale)
        input_mix_1 = T.decl_buffer((num_tokens * 4,), data=input_mix)
        bx = T.launch_thread("blockIdx.x", 20)
        buf_dyn_shmem = T.alloc_buffer((T.int64(1024),), "uint8", scope="shared.dyn")
        with T.sblock("VECTOR", no_realize=True):
            T.reads()
            T.writes()
            with T.sblock("SIMT_VF", no_realize=True):
                T.reads()
                T.writes()
                input_recompute_frag = T.Buffer((1,), scope="local")
                grad_frag = T.Buffer((1,), scope="local")
                mhc_scale_grad_reducer = T.Buffer((1,), scope="local")
                mhc_base_grad_reducer = T.Buffer((1,), scope="local")
                mhc_scale_grad_frag = T.Buffer((1,), scope="local")
                mhc_base_grad_frag = T.Buffer((1,), scope="local")
                T.sblock_attr({"layout_map": {input_recompute_frag: metadata["tl.Fragment"][0], grad_frag: metadata["tl.Fragment"][1], mhc_scale_grad_reducer: metadata["tl.Fragment"][2], mhc_base_grad_reducer: metadata["tl.Fragment"][3], mhc_scale_grad_frag: metadata["tl.Fragment"][4], mhc_base_grad_frag: metadata["tl.Fragment"][5]}, "tl.vf_source_index": T.int64(0)})
                mhc_scale_grad_reducer_1 = T.alloc_buffer((1,), scope="local")
                mhc_scale_grad_reducer_2 = T.decl_buffer((1,), data=mhc_scale_grad_reducer.data, scope="local")
                mhc_base_grad_reducer_1 = T.alloc_buffer((1,), scope="local")
                mhc_base_grad_reducer_2 = T.decl_buffer((1,), data=mhc_base_grad_reducer.data, scope="local")
                mhc_scale_grad_frag_1 = T.alloc_buffer((1,), scope="local")
                mhc_scale_grad_frag_2 = T.decl_buffer((1,), data=mhc_scale_grad_frag.data, scope="local")
                mhc_base_grad_frag_1 = T.alloc_buffer((1,), scope="local")
                mhc_base_grad_frag_2 = T.decl_buffer((1,), data=mhc_base_grad_frag.data, scope="local")
                grad_frag_1 = T.alloc_buffer((1,), scope="local")
                grad_frag_2 = T.decl_buffer((1,), data=grad_frag.data, scope="local")
                input_recompute_frag_1 = T.alloc_buffer((1,), scope="local")
                input_recompute_frag_2 = T.decl_buffer((1,), data=input_recompute_frag.data, scope="local")
                workspace = T.decl_buffer((128,), data=buf_dyn_shmem.data, scope="shared.dyn")
                workspace_1 = T.decl_buffer((128,), data=buf_dyn_shmem.data, scope="shared.dyn")
                simtvf_tx = T.launch_thread("threadIdx.x", 128)
                simtvf_ty = T.launch_thread("threadIdx.y", 1)
                simtvf_tz = T.launch_thread("threadIdx.z", 1)
                T.attr("simtvf", "tl.simtvf_scope", 1)
                mhc_scale_grad_reducer_2[0] = T.float32(0.0)
                mhc_base_grad_reducer_2[0] = T.float32(0.0)
                for w in range((num_tokens + 639) // 640):
                    if w * 20 + bx < (num_tokens + 31) // 32:
                        if w * 640 + bx * 32 + simtvf_tx // 4 < num_tokens:
                            input_recompute_frag_2[0] = T.sigmoid(input_mix_1[T.Cast("int64", w) * T.int64(2560) + T.Cast("int64", bx) * T.int64(128) + T.Cast("int64", simtvf_tx)] * mhc_scale_1[0] + mhc_base_1[simtvf_tx % 4])
                            grad_frag_2[0] = input_recompute_frag_2[0] * (T.float32(1.0) - input_recompute_frag_2[0]) * output_mix_grad_1[T.Cast("int64", w) * T.int64(2560) + T.Cast("int64", bx) * T.int64(128) + T.Cast("int64", simtvf_tx)]
                            input_mix_grad_1[T.Cast("int64", w) * T.int64(2560) + T.Cast("int64", bx) * T.int64(128) + T.Cast("int64", simtvf_tx)] = grad_frag_2[0] * mhc_scale_1[0]
                            mhc_scale_grad_reducer_2[0] = mhc_scale_grad_reducer_2[0] + grad_frag_2[0] * input_mix_1[T.Cast("int64", w) * T.int64(2560) + T.Cast("int64", bx) * T.int64(128) + T.Cast("int64", simtvf_tx)]
                            mhc_base_grad_reducer_2[0] = mhc_base_grad_reducer_2[0] + grad_frag_2[0]
                mhc_scale_grad_reducer_2[0] = T.call_extern("float32", "tl::AscendAllReduce<tl::SumOp, 128, 4, 0>::run", mhc_scale_grad_reducer_2[0], T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem.data, T.FloorDiv(0, 32), 128, 3))
                mhc_scale_grad_reducer_2[0] = T.call_extern("float32", "tl::AscendAllReduce<tl::SumOp, 4, 1, 0>::run", mhc_scale_grad_reducer_2[0])
                mhc_scale_grad_frag_2[0] = mhc_scale_grad_reducer_2[0]
                mhc_base_grad_reducer_2[0] = T.call_extern("float32", "tl::AscendAllReduce<tl::SumOp, 128, 4, 0>::run", mhc_base_grad_reducer_2[0], T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem.data, T.FloorDiv(4096, 32), 128, 3))
                mhc_base_grad_frag_2[0] = mhc_base_grad_reducer_2[0]
                if simtvf_tx == 0:
                    mhc_scale_grad_partial_1[bx] = mhc_scale_grad_frag_2[0]
                if simtvf_tx // 4 == 0:
                    mhc_base_grad_partial_1[bx * 4 + simtvf_tx % 4] = mhc_base_grad_frag_2[0]

# Metadata omitted. Use show_meta=True in script() method to show it.