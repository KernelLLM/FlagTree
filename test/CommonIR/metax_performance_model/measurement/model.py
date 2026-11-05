from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class MeasurementTarget:
    name: str
    llvm: str
    expected_isa: str


@dataclass(frozen=True)
class MeasurementSpec:
    measurement_id: str
    target: MeasurementTarget
    metric: str
    raw_role: str
    independent_chains: int
    input_restrictions: tuple[str, ...]

    conditions: dict

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QualificationResult:
    passed: bool
    reason: str | None
    evidence: dict


@dataclass(frozen=True)
class CostResult:
    run_id: str
    measurement_id: str
    target: dict
    metric: str
    raw_role: str
    independent_chains: int
    input_restrictions: tuple[str, ...]
    value: float
    unit: str
    environment: dict
    native_opcode: str
    conditions: dict
    timing_protocol: str

    def __post_init__(self):
        validate_cost_result(asdict(self))


def load_cost_result(path):
    return CostResult(**read_json(path))


def validate_cost_result(row):
    from .benchmarks import lookup

    if row['timing_protocol'] != 'short_straight_line':
        raise ValueError('short straight-line result required')
    if row['metric'] not in ('independent_issue', 'dependent_latency', 'completion'):
        raise ValueError('unknown measurement metric: ' + row['metric'])
    if not math.isfinite(row['value']) or row['value'] <= 0 or row['unit'] != 'ticks/instruction':
        raise ValueError('cost requires positive finite ticks/instruction')
    spec, _ = lookup(row['measurement_id'])
    for field, expected in spec.to_dict().items():
        actual = row[field]
        if field == 'input_restrictions':
            actual, expected = tuple(actual), tuple(expected)
        if actual != expected:
            raise ValueError(f"{spec.measurement_id}: {field} differs from measurement definition")
    if row['native_opcode'] != spec.target.expected_isa:
        raise ValueError(f"{spec.measurement_id}: native opcode differs")
    environment = row['environment']
    if (environment.get('target') != 'xcore1000' or environment.get('gpu') not in ('MetaX C550', 'MXC550')
            or environment.get('warp_size', 64) != 64):
        raise ValueError(f"{spec.measurement_id}: requires C550/xcore1000/wave64")
