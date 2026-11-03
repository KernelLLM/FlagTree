# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact original device function; preserves int64 program IDs.
Full licensed dependency is retained under tle/dependencies/.
"""
import triton
import triton.language as tl

@triton.jit
def program_id(axis: int) -> tl.tensor:
    return tl.program_id(axis).to(tl.int64)
