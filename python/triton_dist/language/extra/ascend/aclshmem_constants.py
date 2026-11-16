# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from enum import IntEnum

# aclshmem_cmp_op_type_t
class ACLSHMEMCmpOp(IntEnum):
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5

# aclshmem_signal_op_type_t
class ACLSHMEMSignalOp(IntEnum):
    SET = 0
    ADD = 1


# aclshmem_team_t
class ACLSHMEMTeam:
    INVALID = -1
    WORLD = 0


class ACLSHMEMAtomicOp(IntEnum):
    """Compile-time selectors for grouped scalar atomic APIs."""

    NONE = 0
    INC = 1
    ADD = 2
    AND = 3
    OR = 4
    XOR = 5
    SET = 6
    SWAP = 7


class ACLSHMEMTransport(IntEnum):
    """Compile-time API selection; AUTO delegates routing to ACLSHMEM."""

    AUTO = 0
    UDMA = 1
    ROCE = 2
    MTE = 3
