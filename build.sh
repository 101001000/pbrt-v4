#!/usr/bin/env bash
#
# build.sh -- build pbrt-v4 with the portableRT backend.
#
# Notes on the recipe (do not "simplify" these away):
#   * Host code is compiled with g++ only. DPC++ is used exclusively for the
#     SYCL kernel pipeline (auto-detected at configure time through
#     DPCPP_ROOT; without it configure fails with "SYCL was not found").
#   * CUDA must stay OFF (-DPBRT_ENABLE_CUDA=OFF).
#   * Imath is fetched by CMake (FetchContent) but its auto-generated
#     Makefile delegates "all" recursively; it must be built by hand once,
#     after the configure pass, before building the main targets.
#   * pbrt_soa_generated must exist before the kernel preprocessing step
#     runs, otherwise the first kernels.cpp preprocess races on the missing
#     pbrt_soa.h.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build-portablert"
JOBS="${JOBS:-$(nproc)}"

PORTABLERT_ROOT="${PORTABLERT_ROOT:-${ROOT}/../portableRT/build/install}"
DPCPP_ROOT="${DPCPP_ROOT:-/scratch/homelocal/encalle/dpc++6.1.0}"

CMAKE="${CMAKE:-cmake}"

# ---------------------------------------------------------------------------
# 1. Configure (single pass; do not reconfigure after Imath was found,
#    that loses the Imath target)
# ---------------------------------------------------------------------------
"${CMAKE}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DPBRT_BUILD_PORTABLERT=ON \
    -DPBRT_ENABLE_CUDA=OFF \
    -DDPCPP_ROOT="${DPCPP_ROOT}" \
    -DPORTABLERT_ROOT="${PORTABLERT_ROOT}" \
    -S "${ROOT}"


# ---------------------------------------------------------------------------
# 3. SOA code generation (must precede kernel preprocessing: pbrt_soa.h)
# ---------------------------------------------------------------------------
"${CMAKE}" --build "${BUILD_DIR}" --target pbrt_soa_generated -j "${JOBS}"

# ---------------------------------------------------------------------------
# 4. Full build
# ---------------------------------------------------------------------------
"${CMAKE}" --build "${BUILD_DIR}" -j "${JOBS}"

# ---------------------------------------------------------------------------
# 5. PortableRT runtime library on the loader path (for running the binaries
#    manually; this script does not execute anything)
# ---------------------------------------------------------------------------
echo
echo "Build finished. To run pbrt (CPU backend):"
echo "  export LD_LIBRARY_PATH=${PORTABLERT_ROOT}/lib:\$LD_LIBRARY_PATH"
echo "  ${BUILD_DIR}/pbrt [options] scene.pbrt"
