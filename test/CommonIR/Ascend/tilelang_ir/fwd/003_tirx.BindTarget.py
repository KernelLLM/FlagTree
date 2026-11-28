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
        # with T.sblock("root"):
        for bx in T.thread_binding(20, thread="blockIdx.x"):
            tx: T.int32 = T.launch_thread_idx(0)
            ty: T.int32 = T.launch_thread_idx(1)
            tz: T.int32 = T.launch_thread_idx(2)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                input_ub = T.sblock_alloc_buffer((32, 4), scope="shared.dyn")
                w = T.int32()
                pid: T.int32 = T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 32 - 1) // 32))) * T.max(1, T.min(8, (num_tokens + 32 - 1) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 32 - 1) // 32)))
                for w in T.serial((((num_tokens + 32 - 1) // 32 + T.max(1, T.min(8, (num_tokens + 32 - 1) // 32)) - 1) // T.max(1, T.min(8, (num_tokens + 32 - 1) // 32)) * T.max(1, T.min(8, (num_tokens + 32 - 1) // 32)) + 20 - 1) // 20, annotations={"num_stages": 2}):
                    if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 32 - 1) // 32))) * T.max(1, T.min(8, (num_tokens + 32 - 1) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 32 - 1) // 32))) < (num_tokens + 32 - 1) // 32:
                        T.ascend_copy(T.region(input_mix_1[pid * 32, 0], 1, 32, 4), T.region(input_ub[0, 0], 2, 32, 4))
                        with T.sblock("SIMT_VF", no_realize=True):
                            T.reads()
                            T.writes()
                            T.sblock_attr({"tl.vf_source_index": T.int64(0)})
                            input_frag = T.sblock_alloc_buffer((32, 4), scope="local.fragment")
                            simtvf_tx = T.launch_thread("threadIdx.x", 256)
                            simtvf_ty = T.launch_thread("threadIdx.y", 1)
                            simtvf_tz = T.launch_thread("threadIdx.z", 1)
                            T.attr("simtvf", "tl.simtvf_scope", 1)
                            T.ascend_copy(T.region(input_ub[0, 0], 1, 32, 4), T.region(input_frag[0, 0], 2, 32, 4))
                            for i in T.parallel(32):
                                for j in T.parallel(4):
                                    token: T.int32 = pid * 32 + i
                                    if token < num_tokens:
                                        output_mix_1[token, j] = T.sigmoid(input_frag[i, j] * mhc_scale_1[0] + mhc_base_1[j]) + T.float32(9.9999999999999995e-07)