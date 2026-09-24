# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_group5_20260913/dequantize_and_gather_k_cache/original/operator.py
# Provenance: results/gluon/gluon_baseline/no_gather_lens.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
# E1: 64-element vectors are owned by one gfx936 wave64.
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def _dequantize_and_gather_k_cache_kernel_18ccedc7f2(out_ptr, out_stride0, out_stride1, k_cache_ptr, seq_lens_ptr, block_table_ptr, offset, gather_lens_ptr, max_blocks_per_seq: gl.constexpr, nope_dim: gl.constexpr, rope_dim: gl.constexpr, scale_slots: gl.constexpr, quant_block: gl.constexpr, cache_block_size: gl.constexpr, token_data_size: gl.constexpr, cache_block_stride: gl.constexpr, output_dim: gl.constexpr, num_workers: gl.constexpr, n_quant_blocks: gl.constexpr, HAVE_GATHER_LENS: gl.constexpr):
    """HCU helper SHA256: b0cc7a376fae475a54a624352c5f0d47007b582aaa4a6392dad821553c4b4518"""
    req_idx = __h81.freeze(gl.program_id(0))
    worker_idx = __h81.freeze(gl.program_id(1))
    seq_len = __h81.freeze(__h81.load(__h81.to_auto(seq_lens_ptr) + __h81.to_auto(req_idx)))
    if __h81.to_auto(HAVE_GATHER_LENS):
        gather_len = __h81.freeze(__h81.load(__h81.to_auto(gather_lens_ptr) + __h81.to_auto(req_idx)))
    else:
        gather_len = __h81.freeze(__h81.to_auto(seq_len))
    start_pos = __h81.freeze(__h81.to_auto(seq_len) - __h81.to_auto(gather_len))
    for local_i in range(__h81.to_auto(worker_idx), __h81.to_auto(gather_len), __h81.to_auto(num_workers)):
        pos = __h81.freeze(__h81.to_auto(start_pos) + __h81.to_auto(local_i))
        block_in_seq = __h81.freeze(__h81.to_auto(pos) // __h81.to_auto(cache_block_size))
        pos_in_block = __h81.freeze(__h81.to_auto(pos) - __h81.to_auto(block_in_seq) * __h81.to_auto(cache_block_size))
        physical_block = __h81.freeze(__h81.load(__h81.to_auto(block_table_ptr) + __h81.to_auto(req_idx) * __h81.to_auto(max_blocks_per_seq) + __h81.to_auto(block_in_seq)))
        cache_block = __h81.freeze(__h81.to_auto(k_cache_ptr) + __h81.to_auto(physical_block).to(gl.int64) * __h81.to_auto(cache_block_stride))
        token_data = __h81.freeze(__h81.to_auto(cache_block) + __h81.to_auto(pos_in_block) * __h81.to_auto(token_data_size))
        scale_base = __h81.freeze(__h81.to_auto(cache_block) + __h81.to_auto(cache_block_size) * __h81.to_auto(token_data_size) + __h81.to_auto(pos_in_block) * __h81.to_auto(scale_slots))
        out_row = __h81.freeze(__h81.to_auto(out_ptr) + __h81.to_auto(req_idx) * __h81.to_auto(out_stride0) + (__h81.to_auto(offset) + __h81.to_auto(local_i)) * __h81.to_auto(out_stride1))
        if __h81.compare(__h81.to_auto(nope_dim) % __h81.to_auto(quant_block), 0, 'eq'):
            for qblock in gl.static_range(0, __h81.to_auto(n_quant_blocks)):
                qoffs = __h81.freeze(__h81.to_auto(qblock) * __h81.to_auto(quant_block) + __h81.arange(0, __h81.to_auto(quant_block), layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                x_u8 = __h81.freeze(__h81.load(__h81.to_auto(token_data) + __h81.to_auto(qoffs), _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                x_fp8 = __h81.freeze(__h81.to_auto(x_u8).to(gl.float8e4nv, bitcast=True).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                encoded = __h81.freeze(__h81.load(__h81.to_auto(scale_base) + __h81.to_auto(qblock)))
                scale = __h81.freeze(gl.exp2(__h81.to_auto(encoded).to(gl.float32) - 127.0))
                x = __h81.freeze(__h81.to_auto(x_fp8) * __h81.to_auto(scale), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                __h81.store(__h81.to_auto(out_row) + __h81.to_auto(qoffs), __h81.to_auto(x).to(gl.bfloat16), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        else:
            for qblock in gl.static_range(0, __h81.to_auto(n_quant_blocks) - 1):
                qoffs = __h81.freeze(__h81.to_auto(qblock) * __h81.to_auto(quant_block) + __h81.arange(0, __h81.to_auto(quant_block)), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                x_u8 = __h81.freeze(__h81.load(__h81.to_auto(token_data) + __h81.to_auto(qoffs)), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                x_fp8 = __h81.freeze(__h81.to_auto(x_u8).to(gl.float8e4nv, bitcast=True).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                encoded = __h81.freeze(__h81.load(__h81.to_auto(scale_base) + __h81.to_auto(qblock)))
                scale = __h81.freeze(gl.exp2(__h81.to_auto(encoded).to(gl.float32) - 127.0))
                x = __h81.freeze(__h81.to_auto(x_fp8) * __h81.to_auto(scale), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                __h81.store(__h81.to_auto(out_row) + __h81.to_auto(qoffs), __h81.to_auto(x).to(gl.bfloat16))
            qblock = __h81.freeze(__h81.to_auto(n_quant_blocks) - 1)
            qoffs = __h81.freeze(__h81.to_auto(qblock) * __h81.to_auto(quant_block) + __h81.arange(0, __h81.to_auto(quant_block)), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            qmask = __h81.freeze(__h81.compare(__h81.to_auto(qoffs), __h81.to_auto(nope_dim), 'lt'))
            x_u8 = __h81.freeze(__h81.load(__h81.to_auto(token_data) + __h81.to_auto(qoffs), mask=__h81.to_auto(qmask), other=0), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            x_fp8 = __h81.freeze(__h81.to_auto(x_u8).to(gl.float8e4nv, bitcast=True).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            encoded = __h81.freeze(__h81.load(__h81.to_auto(scale_base) + __h81.to_auto(qblock)))
            scale = __h81.freeze(gl.exp2(__h81.to_auto(encoded).to(gl.float32) - 127.0))
            x = __h81.freeze(__h81.to_auto(x_fp8) * __h81.to_auto(scale), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            __h81.store(__h81.to_auto(out_row) + __h81.to_auto(qoffs), __h81.to_auto(x).to(gl.bfloat16), mask=__h81.to_auto(qmask))
        bf16_ptr = __h81.freeze((__h81.to_auto(token_data) + __h81.to_auto(nope_dim)).to(gl.pointer_type(gl.bfloat16)))
        for rblock in gl.static_range(0, __h81.to_auto(rope_dim), 64):
            roffs = __h81.freeze(__h81.to_auto(rblock) + __h81.arange(0, 64))
            rmask = __h81.freeze(__h81.compare(__h81.to_auto(roffs), __h81.to_auto(rope_dim), 'lt'), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            vals = __h81.freeze(__h81.load(__h81.to_auto(bf16_ptr) + __h81.to_auto(roffs), mask=__h81.to_auto(rmask), other=0.0, _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            __h81.store(__h81.to_auto(out_row) + __h81.to_auto(nope_dim) + __h81.to_auto(roffs), __h81.to_auto(vals), mask=__h81.to_auto(rmask), _layout=gl.BlockedLayout([1], [64], [1], [0]))

KERNEL = _dequantize_and_gather_k_cache_kernel_18ccedc7f2
