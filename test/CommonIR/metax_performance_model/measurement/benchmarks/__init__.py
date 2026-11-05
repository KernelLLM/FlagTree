"""Explicit native targets; no source-language discovery or automatic measurement selection."""
from ..model import MeasurementSpec, MeasurementTarget
from . import llvm
from .shared import GEMM_LDS_MAPS, lane_offsets

TARGETS = {
    'fadd_f32': dict(kind='fadd', opcode='add_f32', width=1, llvm='fadd float', rhs=1.), 'fmul_f32':
    dict(kind='fmul', opcode='mul_f32', width=1, llvm='fmul float',
         rhs=1.00390625), 'fma_f32': dict(kind='fma', opcode='fmac_f32', width=1, llvm='llvm.fma.f32',
                                          rhs=1.), 'mma_f16_16x16x16':
    dict(kind='mma', opcode='mma_16x16x16f16', width=4, llvm='llvm.mxc.mma.f32.16x16x16f16', rhs=1.), **{
        name: dict(kind='integer', opcode=opcode, width=1, llvm=op + ' i32', rhs=rhs)
        for name, opcode, op, rhs in (('i32_add.varying', 'add_u32', 'add', 0x12345), ('i32_and', 'and_b32', 'and', 0x5a5aa5a5), ('i32_or', 'or_b32', 'or', 0xc0ffee), ('i32_xor', 'xor_b32', 'xor', 0xdeadbeef), ('shl_i32.imm16', 'shl_b32', 'shl', 16), ('lshr_i32.imm16', 'shr_b32', 'lshr', 16))
    }, **{
        name: dict(kind='perm', opcode='perm_b32', width=1, llvm='llvm.mxc.byte.perm', selector=selector, rhs=0x76543210)
        for name, selector in (('byte_perm.front', 0x01000504), ('byte_perm.back', 0x03020706))
    }, **{
        f'lds_{suffix}':
        dict(kind='lds', opcode=f'lds_{suffix}', width=max(1, bits // 32), size=bits // 8, llvm=f'AS3 load {bits} bits; native output tuple')
        for suffix, bits in (('u16', 16), ('b32', 32), ('b64', 64), ('b128', 128))
    }, **{
        f'sts_b{bits}':
        dict(kind='sts', opcode=f'sts_b{bits}', width=max(1, bits // 32), size=bits // 8, llvm=f'AS3 store {bits} bits; native data tuple')
        for bits in (16, 32, 64, 128)
    }, 'bpermute.i32.xor1': dict(kind='bpermute', opcode='sm_bperm_b32', width=1, size=4,
                                 llvm='llvm.mxc.bsm.bpermute'), **{
                                     f'lds_gemm.map{i}':
                                     dict(kind='lds', opcode='lds_u16' if size == 2 else f'lds_b{size*8}', width=max(
                                         1, size // 4), size=size, llvm=f'AS3 load {size*8} bits; GEMM lane map {i}', address_terms=terms)
                                     for i, (size, terms) in enumerate(GEMM_LDS_MAPS)
                                 }
}


def definition(name, metric):
    t = TARGETS[name]
    kind = t['kind']
    independent = metric == 'independent_issue'
    if metric not in ('independent_issue',
                      'dependent_latency') or not independent and kind in ('lds', 'sts', 'bpermute'):
        raise ValueError('no isolated measurement: ' + name + '.' + metric)
    conditions = {'active_lanes': 64, 'width_words': t['width'], 'mask': 'all', 'operand_binding': kind}
    restrictions = [
        'finite exactly representable test inputs' if kind in ('fadd', 'fmul', 'fma',
                                                               'mma') else 'full-wave integer bit patterns'
    ]
    if kind in ('lds', 'sts'):
        conditions.update(address_space=3, alignment=t['size'], access_bytes=t['size'],
                          address_pattern='eight_disjoint_slabs', slabs=8, volatile=True)
        restrictions = [
            'AS3; address(slot,lane)=(slot*64+lane)*access_bytes; fixed slab per target; no concurrent writes',
            'no in-window wait or consumer; post-stop BSM recovery'
        ]
        if 'address_terms' in t:
            conditions.pop('volatile')
            conditions.update(address_pattern='gemm_lane_map', lane_byte_offsets=lane_offsets(t['address_terms']),
                              slot_stride=4096)
            restrictions = [
                'AS3; eight disjoint 4096-byte slots; exact lane offsets; natural alignment; no concurrent writes',
                'native LDS issue only; LLVM volatile is not an ISA modifier; readiness, base translation and interleaving not measured'
            ]
    elif kind == 'bpermute':
        conditions.update(address_pattern='xor1', modifier=0)
        restrictions = [
            'byte address=(lane xor 1)*4; full wave64; independent source/result slots; post-stop BSM recovery'
        ]
    elif kind == 'perm':
        conditions.update(selector=t['selector'], invariant=t['rhs'])
    elif kind == 'integer':
        conditions.update(invariant=t['rhs'], register_class='MTREG', immediate='imm16' in name)
    else:
        conditions['invariant'] = t['rhs']
        restrictions.append('A/B=1 invariant' if kind in ('fma', 'mma') else 'invariant RHS=' + str(t['rhs']))
    raw = 'none; invariant address and data' if kind == 'sts' else 'independent complete result tuples' if independent else 'C accumulator tuple' if kind in (
        'fma', 'mma') else 'first data operand feedback'
    return MeasurementSpec(name + '.' + metric, MeasurementTarget(name, t['llvm'], t['opcode']), metric, raw,
                           (4 if kind == 'bpermute' else 8) if independent else 1, tuple(restrictions), conditions)


def list_measurements():
    return [
        definition(name, m)
        for name, t in TARGETS.items()
        for m in (('independent_issue', ) if t['kind'] in ('lds', 'sts', 'bpermute') else ('independent_issue',
                                                                                           'dependent_latency'))
    ]


def lookup(measurement_id):
    for spec in list_measurements():
        if spec.measurement_id == measurement_id: return spec, llvm
    raise ValueError('unknown measurement: ' + measurement_id)
