"""Generic model context from launch metadata, without operator/name adapters.

ABI positions and scalar values are binding/provenance only. The native producer
derives physical scale facts; raw argument values are never predictor columns.
"""
import json
import math
import re

SCHEMA = 'hcu-layout-execution-context-v1'
OPTION = 'layout_execution_context'
MAX_EXACT = 2**53 - 1


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def canonical_grid(grid):
    if grid is None:
        return None
    if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3:
        raise ValueError('model context needs a one to three dimensional launch grid')
    values = list(grid) + [1] * (3 - len(grid))
    if any(type(x) is not int or not 0 <= x <= 2**31 - 1 for x in values):
        raise ValueError('model context grid must contain nonnegative i32 integers')
    if math.prod(values) > MAX_EXACT:
        raise ValueError('model context CTA count exceeds exact feature range')
    return values


def _get(args, path):
    value = args[path[0]]
    for index in path[1:]:
        if _is_descriptor(value):
            value = _descriptor_fields(value)[index]
        else:
            value = value[index]
    return value


def _is_descriptor(value):
    return all(hasattr(value, name) for name in ('base', 'shape', 'strides', 'block_shape', 'padding'))


def _descriptor_fields(value, block_shape=None):
    """Match HCU's descriptor-to-pointer pass and launcher, including duplicate ports.

    This context is enabled by the HCU backend, which rewrites descriptors before
    the layout slot. Opaque hardware descriptor ABIs need a separate contract.
    """
    if not _is_descriptor(value):
        raise ValueError('model descriptor binding changed type')
    shape, strides, block = list(value.shape), list(value.strides), list(value.block_shape)
    rank = len(shape)
    if (not 1 <= rank <= 5 or len(strides) != rank or len(block) != rank or
            any(type(x) is not int or not 0 < x < 2**31 for x in shape) or
            any(type(x) is not int or not 0 <= x <= MAX_EXACT for x in strides) or
            any(type(x) is not int or x <= 0 or x & (x - 1) for x in block) or
            strides[-1] != 1 or value.padding not in ('zero', 'nan') or
            (block_shape is not None and block != block_shape)):
        raise ValueError('invalid descriptor metadata for decomposed HCU ABI')
    return [value.base, *shape, *strides, value.padding == 'nan', *shape, *strides]


def _tensor(value):
    if not all(hasattr(value, name) for name in ('shape', 'stride', 'element_size')):
        return None
    shape = [int(x) for x in value.shape]
    strides = [int(x) for x in value.stride()]
    size = int(value.element_size())
    if len(shape) != len(strides) or len(shape) > 32 or size not in (1, 2, 4, 8, 16):
        raise ValueError('invalid tensor metadata in model execution context')
    if any(x < 0 or x > MAX_EXACT for x in shape + strides):
        raise ValueError('tensor metadata exceeds supported exact nonnegative range')
    span = 0 if not math.prod(shape) else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    if max(math.prod(shape), span) * size > MAX_EXACT:
        raise ValueError('tensor byte extent exceeds exact feature range')
    return dict(shape=shape, strides=strides, element_bytes=size)


def _snapshot(template, values):
    row = {k: v for k, v in template.items() if k not in ('value', 'tensor')}
    value = _get(values, row['path'])
    kind = row['kind']
    if kind == 'integer':
        bits, unsigned = row['bits'], row['unsigned']
        if type(value) not in (int, bool):
            raise ValueError('integer model binding changed type')
        value = int(value)
        lo, hi = (0, 2**bits - 1) if unsigned else (-(2**(bits-1)), 2**(bits-1)-1)
        if not lo <= value <= hi or abs(value) > MAX_EXACT:
            raise ValueError('integer model binding outside exact ABI range')
        row['value'] = value
    elif kind == 'pointer':
        row['tensor'] = _tensor(value)
    return row


def bind_context(bound_args, specialization, grid):
    values = list(bound_args.values())
    arguments = []

    def visit(spec, path):
        # JIT specialization trees store (type, attributes) at scalar leaves.
        if not (isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str)):
            for index, item in enumerate(spec):
                visit(item, path + [index])
            return
        typename = spec[0]
        if typename == 'constexpr':
            return
        if typename.startswith('tensordesc'):
            match = re.fullmatch(r'tensordesc<([^\[\]<>]+)\[([0-9, ]+)\]>', typename)
            if not match:
                raise ValueError('unsupported descriptor specialization in model context')
            block = [int(x.strip()) for x in match[2].split(',')]
            _descriptor_fields(_get(values, path), block)
            rank = len(block)
            leaves = ['*' + match[1]] + ['i64'] * (2 * rank) + ['i1'] + ['i32'] * rank + ['i64'] * rank
            for index, leaf in enumerate(leaves):
                visit((leaf, None), path + [index])
            return
        integer = re.fullmatch(r'([iu])(1|8|16|32|64)', typename)
        row = dict(index=len(arguments), path=path,
                   kind='integer' if integer else 'pointer' if typename.startswith('*') else 'other')
        if integer:
            row.update(bits=int(integer[2]), unsigned=integer[1] == 'u' or integer[2] == '1')
        arguments.append(_snapshot(row, values))

    for index, item in enumerate(specialization):
        visit(item, [index])
    payload = _encode(dict(schema=SCHEMA, origin='jit.bound_args+grid',
                           launch_arity=len(values), grid=canonical_grid(grid), arguments=arguments))
    parse_context(payload)
    return payload


def parse_context(payload):
    if type(payload) is not str or len(payload) > 65536:
        raise ValueError('invalid model execution context serialization')
    doc = json.loads(payload)
    if (type(doc) is not dict or set(doc) != {'schema', 'origin', 'launch_arity', 'grid', 'arguments'}
            or doc['schema'] != SCHEMA or doc['origin'] != 'jit.bound_args+grid'
            or type(doc['launch_arity']) is not int or not 0 <= doc['launch_arity'] <= 1024
            or type(doc['arguments']) is not list or len(doc['arguments']) > 1024):
        raise ValueError('invalid generic model execution context')
    if canonical_grid(doc['grid']) != doc['grid']:
        raise ValueError('model context grid is not canonical')
    paths = set()
    for index, row in enumerate(doc['arguments']):
        if type(row) is not dict or row.get('index') != index or row.get('kind') not in ('integer', 'pointer', 'other'):
            raise ValueError('invalid model context ABI binding')
        path = row.get('path')
        if (type(path) is not list or not 1 <= len(path) <= 32 or
                any(type(x) is not int or not 0 <= x < 1024 for x in path) or
                path[0] >= doc['launch_arity'] or tuple(path) in paths):
            raise ValueError('invalid model context argument path')
        paths.add(tuple(path))
        common = {'index', 'path', 'kind'}
        if row['kind'] == 'integer':
            if (set(row) != common | {'bits', 'unsigned', 'value'} or
                    row['bits'] not in (1, 8, 16, 32, 64) or type(row['unsigned']) is not bool or
                    type(row['value']) is not int):
                raise ValueError('invalid model context integer')
            lo, hi = (0, 2**row['bits'] - 1) if row['unsigned'] else (-(2**(row['bits']-1)), 2**(row['bits']-1)-1)
            if not lo <= row['value'] <= hi or abs(row['value']) > MAX_EXACT:
                raise ValueError('model context integer exceeds exact ABI range')
        elif row['kind'] == 'pointer':
            if set(row) != common | {'tensor'}:
                raise ValueError('invalid model context pointer')
            t = row['tensor']
            if t is not None:
                if type(t) is not dict or set(t) != {'shape', 'strides', 'element_bytes'}:
                    raise ValueError('invalid model context tensor')
                class Metadata:
                    shape = t['shape']
                    def stride(self): return t['strides']
                    def element_size(self): return t['element_bytes']
                if _tensor(Metadata()) != t:
                    raise ValueError('noncanonical model tensor metadata')
        elif set(row) != common:
            raise ValueError('invalid model context opaque argument')
    if _encode(doc) != payload:
        raise ValueError('model execution context is not canonical')
    return doc


def validate_launch(payload, args, grid):
    doc = parse_context(payload)
    if len(args) != doc['launch_arity'] or canonical_grid(list(grid)) != doc['grid']:
        raise ValueError('model execution context launch/grid mismatch')
    if [_snapshot(row, args) for row in doc['arguments']] != doc['arguments']:
        raise ValueError('model execution context launch argument metadata mismatch')


def rebind_compiled_kernel(kernel, grid, args, backend=None):
    """Specialize a host-cached kernel again when its actual workload changes.

    Custom host caches may reuse a CompiledKernel across grids or tensor
    metadata. Keep the low-level guard strict, and use the ordinary JIT cache
    to obtain a model-scored specialization for this complete launch binding.
    """
    payload = getattr(kernel.metadata, OPTION, '')
    if not payload:
        return None
    doc = parse_context(payload)
    if len(args) != doc['launch_arity']:
        raise ValueError('model execution context launcher ABI mismatch')
    actual_grid = canonical_grid(list(grid))
    if (actual_grid == doc['grid'] and
            [_snapshot(row, args) for row in doc['arguments']] == doc['arguments']):
        return None
    source = kernel.src
    if not hasattr(source, 'fn') or len(source.fn.arg_names) != len(args):
        raise ValueError('model execution context requires matching ASTSource binding')
    if backend is None:
        from triton.compiler import make_backend
        backend = make_backend(kernel.metadata.target)
    if (not getattr(backend, 'uses_layout_execution_context', lambda: False)() or
            backend.hash() != kernel._layout_rebind_backend_hash):
        raise ValueError('model execution context cannot rebind under a different compile policy')
    from dataclasses import fields
    options = {field.name: getattr(kernel.metadata, field.name)
               for field in fields(type(backend.parse_options({})))
               if hasattr(kernel.metadata, field.name) and
               field.name not in ('layout_execution_context',)}
    arguments = dict(zip(source.fn.arg_names, args))
    for name, value in options.items():
        if name in arguments and arguments[name] != value:
            raise ValueError('source argument and compiler option disagree: ' + name)
        arguments[name] = value
    # warmup=True compiles or finds a specialization without launching it.
    rebound = source.fn.run(grid=actual_grid, warmup=True, **arguments)
    if rebound is None:
        raise ValueError('model execution context recompilation returned no kernel')
    if getattr(rebound, '_layout_rebind_backend_hash', None) != kernel._layout_rebind_backend_hash:
        raise ValueError('model execution context JIT cache contains a different compile policy')
    validate_launch(getattr(rebound.metadata, OPTION, ''), args, actual_grid)
    return rebound


def guard_launcher(launcher, source, payload):
    doc = parse_context(payload)
    if not hasattr(source, 'fn') or len(source.fn.arg_names) != doc['launch_arity']:
        raise ValueError('model execution context requires matching ASTSource binding')
    def checked(*args):
        if len(args) != 9 + doc['launch_arity']:
            raise ValueError('model execution context launcher ABI mismatch')
        validate_launch(payload, args[9:], args[:3])
        return launcher(*args)
    return checked
