#!/usr/bin/env bash
# Copyright 2026- Xcoresigma Technology Co., Ltd
# Manual rebuild of custom_ops.<arch>.bc. The CMake build also drives ccec/llvm-link
# (third_party/ascend/CMakeLists.txt) when the CANN toolchain is available;
# this script is the fallback for machines where it is not found at configure
# time, or for rebuilding after editing a device source.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# ``custom_ops.<arch>.bc`` must be built for the same ccec arch family as the kernel:
# c220 for A2/A3, c310 for A5/Ascend950. Export CUSTOM_OPS_ARCH_FAMILY=c310
# when building for a 950 target.
ARCH_FAMILY="${CUSTOM_OPS_ARCH_FAMILY:-c220}"
case "${ARCH_FAMILY}" in
  c220|c310) ;;
  *)
    echo "error: unsupported CUSTOM_OPS_ARCH_FAMILY='${ARCH_FAMILY}' (expected c220 or c310)" >&2
    exit 1
    ;;
esac
CUSTOM_OPS_BC="${SCRIPT_DIR}/custom_ops.${ARCH_FAMILY}.bc"
TEMPLATE_ROOT="${SCRIPT_DIR}/../../../../../../../../third_party/ascend/AscendNPU-IR/bishengir/lib/Template"
TEMPLATE_INCLUDE="${TEMPLATE_ROOT}/include"
TEMPLATE_LIB="${TEMPLATE_ROOT}/lib"
TEMPLATE_BC=(
  "${SCRIPT_DIR}/template_arange.bc"
  "${SCRIPT_DIR}/template_eltwise1d.bc"
  "${SCRIPT_DIR}/template_sort1d.bc"
  "${SCRIPT_DIR}/template_copy1d.bc"
)

# Keep CCEC/LLVM_LINK from the same CANN/BiShengIR release as the
# ``bishengir-compile`` used by Triton; mixing releases can produce
# "Cannot select" intrinsic or data-layout errors when linking device bitcode.
CCEC="${CCEC:-$(command -v ccec || true)}"
if [[ -z "${CCEC}" ]]; then
  echo "error: ccec not found; source the CANN environment first" >&2
  exit 1
fi
CCEC="$(readlink -f "${CCEC}")"

CCEC_BIN_DIR="$(dirname -- "${CCEC}")"
LLVM_LINK="${LLVM_LINK:-${CCEC_BIN_DIR}/llvm-link}"
if [[ ! -x "${LLVM_LINK}" ]]; then
  echo "error: CANN llvm-link not found: ${LLVM_LINK}" >&2
  exit 1
fi

# Each custom op is implemented in its own .cpp and compiled into its own
# bitcode: "<src>::<arch>" — arch is the ccec aicore target for that op.
CUSTOM_OPS=(
  "mem_ops/duplicate.cpp:dav-${ARCH_FAMILY}-vec"
  "mem_ops/gather_gm_to_l1.cpp:dav-${ARCH_FAMILY}-cube"
  "mem_ops/gather_gm_to_ub.cpp:dav-${ARCH_FAMILY}-vec"
  "mem_ops/gather_mask.cpp:dav-${ARCH_FAMILY}-vec"
  "reduction_ops/pair_reduce_sum.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/sort32.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/sort_1d_pack.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/merge_pack_sort.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/mrgsort.cpp:dav-${ARCH_FAMILY}-vec"
  "sort_ops/unpack_sort.cpp:dav-${ARCH_FAMILY}-vec"
  "mask_ops/compare_scalar.cpp:dav-${ARCH_FAMILY}-vec"
  "cast_ops/cast_int4_to_fp16.cpp:dav-${ARCH_FAMILY}-vec"
)

if [[ ! -d "${TEMPLATE_INCLUDE}" ]]; then
  echo "error: Ascend Template include directory not found: ${TEMPLATE_INCLUDE}" >&2
  exit 1
fi

OP_BC=()
for entry in "${CUSTOM_OPS[@]}"; do
  src="${entry%%:*}"
  arch="${entry##*:}"
  if [[ ! -f "${SCRIPT_DIR}/${src}" ]]; then
    echo "error: custom-op source not found: ${SCRIPT_DIR}/${src}" >&2
    exit 1
  fi

  CCEC_COMMON_ARGS=( -O2 -x cce --cce-auto-sync=off --cce-aicore-only
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns
    --cce-aicore-arch="${arch}" --cce-enable-print
    --cce-enable-sanitizer -std=c++17 -I "${TEMPLATE_INCLUDE}" )

  bc="${SCRIPT_DIR}/${src%.cpp}.bc"
  mix_bc="${SCRIPT_DIR}/${src%.cpp}.mix.bc"
  OP_BC+=("${bc}" "${mix_bc}")

  echo "== ccec ${src} (${arch})"
  "${CCEC}" "${CCEC_COMMON_ARGS[@]}" \
    "${SCRIPT_DIR}/${src}" -emit-llvm -c -o "${bc}"

  echo "== ccec ${src} (${arch}, mix)"
  "${CCEC}" "${CCEC_COMMON_ARGS[@]}" -cce-enable-mix -mllvm -enable-mix=true \
    "${SCRIPT_DIR}/${src}" -emit-llvm -c -o "${mix_bc}"
done

TEMPLATE_SOURCES=(
  "${TEMPLATE_LIB}/Vector/Arange/Arange1D.cpp"
  "${TEMPLATE_LIB}/Vector/Elementwise/EltWise1D.cpp"
  "${TEMPLATE_LIB}/Vector/Sort/Sort1D.cpp"
  "${TEMPLATE_LIB}/DMA/Ubuf/Copy1D.cpp"
)
TEMPLATE_MIX_BC=()
for i in "${!TEMPLATE_SOURCES[@]}"; do
  echo "== ccec ${TEMPLATE_SOURCES[$i]}"
  "${CCEC}" -O2 -x cce --cce-auto-sync=off --cce-aicore-only \
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns \
    --cce-aicore-arch=dav-${ARCH_FAMILY}-vec --cce-enable-print \
    --cce-enable-sanitizer -std=c++17 -I "${TEMPLATE_INCLUDE}" \
    "${TEMPLATE_SOURCES[$i]}" -emit-llvm -c -o "${TEMPLATE_BC[$i]}"
  mix_bc="${TEMPLATE_BC[$i]%.bc}.mix.bc"
  echo "== ccec ${TEMPLATE_SOURCES[$i]} (mix)"
  "${CCEC}" -O2 -x cce --cce-auto-sync=off --cce-aicore-only \
    --cce-generic-addrspace=off -mllvm -disable-llvm-optzns \
    --cce-aicore-arch=dav-${ARCH_FAMILY}-vec --cce-enable-print \
    --cce-enable-sanitizer -std=c++17 -I "${TEMPLATE_INCLUDE}" \
    -cce-enable-mix -mllvm -enable-mix=true \
    "${TEMPLATE_SOURCES[$i]}" -emit-llvm -c -o "${mix_bc}"
  TEMPLATE_MIX_BC+=("${mix_bc}")
done

echo "== llvm-link -> ${CUSTOM_OPS_BC}"
"${LLVM_LINK}" "${OP_BC[@]}" "${TEMPLATE_BC[@]}" "${TEMPLATE_MIX_BC[@]}" \
  -o "${CUSTOM_OPS_BC}"

if [[ ! -s "${CUSTOM_OPS_BC}" ]]; then
  echo "error: bitcode was not generated: ${CUSTOM_OPS_BC}" >&2
  exit 1
fi

rm -f "${OP_BC[@]}" "${TEMPLATE_BC[@]}" "${TEMPLATE_MIX_BC[@]}"
stat --format='%n %s bytes' "${CUSTOM_OPS_BC}"
sha256sum "${CUSTOM_OPS_BC}"
