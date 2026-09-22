# portableRT backend for pbrt-v4

This documents the portableRT backend integration: how to build it, how to
evaluate it, current progress, and known limitations.

## Overview

The portableRT backend consumes the *installed* portableRT package
(`../portableRT/build/install`; version 0.4.1, SYCL feature). portableRT is
never modified, built, or reinstalled by this project. The backend reuses the
existing CUDA-oriented wavefront renderer code: the same work queues, work
items, globals, and kernels are executed through portableRT's
backend-agnostic kernel API instead of CUDA/OptiX.

Target configuration: **SYCL (Intel GPUs) and CPU backends with the
`native_bvh2` BVH layout**. All other portableRT backends and layouts are out
of scope.

## Build configurations

| Build dir | CUDA | portableRT | Purpose |
|---|---|---|---|
| `build-cpu-reference/` | OFF | OFF | Frozen CPU baseline executable and reference images |
| `build-portablert/` | OFF | ON | The portableRT integration build |

### CPU reference (baseline)

```sh
cmake -B build-cpu-reference -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu-reference -j 16
```

### portableRT-enabled

```sh
cmake -B build-portablert -DCMAKE_BUILD_TYPE=Release -DPBRT_BUILD_PORTABLERT=ON
cmake --build build-portablert -j 16
```

Options:

- `PBRT_BUILD_PORTABLERT` (default `OFF`): enables the portableRT backend.
  It implies `PBRT_ENABLE_CUDA=OFF`: the CUDA toolchain is not detected, no
  source file is compiled with CUDA, and no OptiX PTX is generated.
- `PBRT_ENABLE_CUDA` (default `ON`): explicit CUDA opt-out independent from
  portableRT.
- `PORTABLERT_ROOT`: path to the installed portableRT package. Defaults to
  `<source>/../portableRT/build/install`.
- `DPCPP_ROOT`: oneAPI DPC++ installation used by portableRT's SYCL kernel
  pipeline. Auto-detected; must contain `bin/clang++`, `lib/libsycl.so` and
  `lib/libur_loader.so` (mirroring portableRT's own `detect_sycl`).

Only the SYCL feature of the installed package is consumed; CUDA, OptiX, HIP,
HIPRT, Embree and Embree-SYCL features are explicitly disabled in
`cmake/portablert.cmake`. The SYCL kernel target is pinned to `spir64`
(zebin `acm-g10` for the installed Intel Arc A770).

## Running

Ordinary CPU rendering, including the CPU wavefront path, works in both
builds:

```sh
./build-cpu-reference/pbrt_exe/pbrt --wavefront scene.pbrt
```

The portableRT backend is selected with the usual `--gpu` flag (see status
below).

## Unit tests

```sh
cmake --build build-cpu-reference -j 16 --target pbrt_test
./build-cpu-reference/pbrt_test/pbrt_test
```

## Status

- [x] C01: build integration (installed portableRT discovery, link, CUDA disabled)
- [ ] C02: runtime initialization through the existing GPU path
- [ ] C03: memory allocation and transfer adapters
- [ ] C04: device globals preserved and initialized
- [ ] C05: single-source kernel launch integration (feasibility gate)
- [ ] C06-C14: wavefront stages, intersections, textures, evaluation

## Recorded results

### C01

- `build-cpu-reference`: clean configure/build, no CUDA compilation (no CUDA
  toolchain available and CUDA detection explicitly skipped).
- `pbrt_test`: all tests pass (see `build-cpu-reference/Testing/`).
- `build-portablert`: configures with `find_package(portableRT CONFIG
  REQUIRED)`, links `portableRT::portableRT`, reports
  `SYCL: ON, DPC++: /scratch/homelocal/encalle/dpc++6.1.0`.
- Build log contains no CUDA compilation and no OptiX PTX generation.
- Baseline CPU wavefront reference images: `build-cpu-reference/reference/`.
