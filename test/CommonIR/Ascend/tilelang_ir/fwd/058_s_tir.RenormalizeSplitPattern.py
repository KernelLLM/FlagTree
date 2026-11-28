# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def mhc_head_compute_mix_fwd_asc(input_mix: T.handle, mhc_scale: T.handle, mhc_base: T.handle, output_mix: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "launch_grid_only": True, "max_vector_bits": 64, "supports_kernel_status_return": False, "supports_vector_predicate": True, "tag": ""}), "tl.has_tma": T.bool(False)})
        num_tokens = T.int32()
        input_mix_1 = T.match_buffer(input_mix, (num_tokens, 4), strides=(4, 1))
        mhc_scale_1 = T.match_buffer(mhc_scale, (1,), strides=(1,))
        mhc_base_1 = T.match_buffer(mhc_base, (4,), strides=(1,))
        output_mix_1 = T.match_buffer(output_mix, (num_tokens, 4), strides=(4, 1))
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `input_mix`, `output_mix`")
        bx = T.launch_thread("blockIdx.x", 20)
        input_ub = T.alloc_buffer((256,), scope="shared.dyn")
        input_ub_1 = T.decl_buffer((256,), data=input_ub.data, scope="shared.dyn")
        input_ub_version_counter_1 = T.alloc_buffer((1,), "int32", scope="local.var")
        input_ub_version_counter_1_1 = T.decl_buffer((1,), "int32", data=input_ub_version_counter_1.data, scope="local.var")
        T.attr({input_ub.data: []}, "tl.buffer_alias_map", 1)
        with T.sblock("VECTOR", no_realize=True):
            T.reads()
            T.writes()
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_set_flag("V_MTE2", 1)
            input_ub_version_counter_1_1[0] = 0
            for w in range(((((num_tokens + 31) // 32 - 1) // T.max(1, T.min(8, (num_tokens + 31) // 32)) + 1) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + 19) // 20):
                __cond_0: T.bool = T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32
                if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32:
                    T.ascend_wait_flag("V_MTE2", input_ub_version_counter_1_1[0] % 2)
                if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32:
                    T.ascend_copy_gm_to_ubuf(T.tvm_access_ptr(T.type_annotation("float32"), input_ub.data, input_ub_version_counter_1_1[0] % 2 * 128, T.min(32, T.max(num_tokens - T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 - T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32, 0)) * 4, 2), T.tvm_access_ptr(T.type_annotation("float32"), input_mix_1.data, T.Div(T.Cast("int64", w) * T.int64(20) + T.Cast("int64", bx), T.max(T.int64(1), T.min(T.int64(8), (T.Cast("int64", num_tokens) + T.int64(31)) // T.int64(32)))) * T.max(T.int64(1), T.min(T.int64(8), (T.Cast("int64", num_tokens) + T.int64(31)) // T.int64(32))) * T.int64(128) + T.truncmod(T.Cast("int64", w) * T.int64(20) + T.Cast("int64", bx), T.max(T.int64(1), T.min(T.int64(8), (T.Cast("int64", num_tokens) + T.int64(31)) // T.int64(32)))) * T.int64(128), T.min(32, T.max(num_tokens - T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 - T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32, 0)) * 4, 1), 0, 1, T.min(32, T.max(num_tokens - T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 - T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32, 0)) * 16, 0, 0, 0, 0, T.min(32, T.max(num_tokens - T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 - T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32, 0)) * 16, T.min(32, T.max(num_tokens - T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 - T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32, 0)) * 16)
                if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32:
                    T.ascend_set_flag("MTE2_V", input_ub_version_counter_1_1[0] % 2)
                if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32:
                    T.ascend_wait_flag("MTE2_V", input_ub_version_counter_1_1[0] % 2)
                if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32:
                    input_frag = T.Buffer((1,), scope="local")
                    with T.sblock("SIMT_VF", no_realize=True):
                        T.reads()
                        T.writes()
                        T.sblock_attr({"layout_map": {input_frag: metadata["tl.Fragment"][0]}, "tl.vf_source_index": T.int64(0)})
                        input_frag_1 = T.alloc_buffer((1,), scope="local")
                        input_frag_2 = T.decl_buffer((1,), data=input_frag.data, scope="local")
                        simtvf_tx = T.launch_thread("threadIdx.x", 256)
                        simtvf_ty = T.launch_thread("threadIdx.y", 1)
                        simtvf_tz = T.launch_thread("threadIdx.z", 1)
                        T.attr("simtvf", "tl.simtvf_scope", 1)
                        input_frag_2[0] = input_ub_1[input_ub_version_counter_1_1[0] % 2 * 128 + simtvf_tx % 128]
                        if simtvf_tx // 128 == 0:
                            if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) * 32 + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * 32 + simtvf_tx // 4 % 32 < num_tokens:
                                output_mix_2 = T.Buffer((num_tokens * 4,), data=output_mix_1.data)
                                mhc_scale_2 = T.Buffer((1,), data=mhc_scale_1.data)
                                mhc_base_2 = T.Buffer((4,), data=mhc_base_1.data)
                                output_mix_2[T.Div(T.Cast("int64", w) * T.int64(20) + T.Cast("int64", bx), T.max(T.int64(1), T.min(T.int64(8), (T.Cast("int64", num_tokens) + T.int64(31)) // T.int64(32)))) * T.max(T.int64(1), T.min(T.int64(8), (T.Cast("int64", num_tokens) + T.int64(31)) // T.int64(32))) * T.int64(128) + T.truncmod(T.Cast("int64", w) * T.int64(20) + T.Cast("int64", bx), T.max(T.int64(1), T.min(T.int64(8), (T.Cast("int64", num_tokens) + T.int64(31)) // T.int64(32)))) * T.int64(128) + T.Cast("int64", simtvf_tx) % T.int64(128)] = T.sigmoid(input_frag_2[0] * mhc_scale_2[0] + mhc_base_2[simtvf_tx % 4]) + T.float32(9.9999999999999995e-07)
                if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32:
                    T.ascend_set_flag("V_MTE2", input_ub_version_counter_1_1[0] % 2)
                if T.Div(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) * T.max(1, T.min(8, (num_tokens + 31) // 32)) + T.truncmod(w * 20 + bx, T.max(1, T.min(8, (num_tokens + 31) // 32))) < (num_tokens + 31) // 32:
                    input_ub_version_counter_1_1[0] = input_ub_version_counter_1_1[0] + 1
            T.ascend_wait_flag("V_MTE2", 0)
            T.ascend_wait_flag("V_MTE2", 1)

# Metadata omitted. Use show_meta=True in script() method to show it.