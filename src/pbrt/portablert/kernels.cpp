// Feasibility probe for the portableRT backend: pbrt's generic callable-based
// launches (GPUParallelFor-style lambdas) are executed through portableRT's
// named-kernel interface.
//
// Each wavefront launch site becomes a named prt_kernel taking one
// trivially-copyable task struct that mirrors the original lambda captures.
// The host side packs the task struct into a byte buffer and calls
// parallel_invoke with the kernel name (see portablert_test.cpp).
//
// A function-pointer bridge was evaluated and rejected: the SYCL compiler
// rejects calls through function pointers, so portable heterogeneous
// execution requires the named-kernel mechanism.
#include <portableRT/portableRT.hpp>
#include <portableRT/kernel/device_common.hpp>

#include <cstdio>
#include <vector>

// ---- Registered globals (storage owned by the host) --------------------
// The generated kernel wrappers access these through portableRT's propagated
// KernelGlobals view; their host-side definitions provide the storage that
// global_copy_to uploads from.
extern float gl_a;
extern int gl_table[64];

// Pointed-to options living in shared memory: pbrt's registered "Options"
// global holds a pointer that device code dereferences (see
// CopyOptionsToGPU). prt_probe_options mirrors that pattern for the tests.
struct ProbeOptions {
    int useGPU;
    float scale;
};
extern ProbeOptions *prt_probe_options;

// ---- Named kernel with a trivially-copyable task struct ----------------
struct ScaleTask {
    const float *in;
    float *out;
    float factor;
    int *table; // exercised via args as well as via the propagated global
};

prt_kernel void scale_task(ScaleTask t) {
    t.out[global_idx] = t.factor * t.in[global_idx] + gl_a;
    // globals: runtime-indexed writes through the propagated view, plus the
    // same write through an argument pointer for comparison (one writer per
    // element keeps both patterns data race free)
    if (global_idx < 64)
        gl_table[global_idx] = (int)global_idx + 100;
    if (global_idx < 8)
        t.table[global_idx] = (int)global_idx;
}

// Registered-pointer global dereference: the blob holds a pointer into
// shared memory, exactly like pbrt's Options handling.
struct OptionsProbeTask {
    int *result;
};

prt_kernel void options_probe(OptionsProbeTask t) {
    if (global_idx < 8)
        t.result[global_idx] = prt_probe_options->useGPU;
}

// Intersection kernel: links the __traversable machinery into this TU. pbrt
// wavefront kernels always intersect, so this reflects the real usage.
prt_kernel void probe_intersect(prt::Ray ray, unsigned char *out) {
    out[global_idx] = prt::closest_hit(ray).valid;
}
