// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// Focused portableRT integration tests.
//
// pbrt's wavefront launches are generic C++ callables executed through three
// adapter boundaries (WavefrontPathIntegrator::ParallelFor, ::Do and
// ForAllQueued). portableRT requires named kernels, so each launch site
// becomes a named prt_kernel taking one trivially-copyable task struct that
// mirrors the original lambda captures. The host side packs the task struct
// into a byte buffer and calls parallel_invoke with the kernel name.
//
// These tests validate the mechanism end to end: task struct serialization,
// invocation index coverage (including sizes that are not multiples of any
// launch block), and access to portableRT-propagated globals.
// See kernels.cpp for the device side.
#include <portableRT/portableRT.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Registered kernel globals (portableRT propagates accesses in the kernel TU;
// the host uploads values through global_copy_to).
float gl_a;
int gl_table[64];

struct ScaleTask {
    const float *in;
    float *out;
    float factor;
    int *table;
};

struct ProbeOptions {
    int useGPU;
    float scale;
};

struct OptionsProbeTask {
    int *result;
};

static int failures = 0;

static void Check(const char *what, bool ok) {
    printf("%-60s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

static bool TestNamedKernel(prt::Backend *backend, size_t N) {
    std::vector<float> in(N), out(N, 0.f);
    for (size_t i = 0; i < N; ++i)
        in[i] = float(i);

    gl_a = 2.5f;
    memset(gl_table, 0, sizeof(gl_table));

    float *d_in = reinterpret_cast<float *>(backend->device_malloc(N * sizeof(float)));
    float *d_out = reinterpret_cast<float *>(backend->device_malloc(N * sizeof(float)));
    int *d_table = reinterpret_cast<int *>(backend->device_malloc(8 * sizeof(int)));
    backend->device_copy_to(d_in, in.data(), N * sizeof(float));
    backend->device_copy_to(d_out, out.data(), N * sizeof(float));
    backend->device_copy_to(d_table, gl_table, 8 * sizeof(int));
    backend->global_copy_to("gl_a", &gl_a, sizeof(float));
    backend->global_copy_to("gl_table", gl_table, sizeof(gl_table));

    // Bounds are guarded inside the kernel, so sizes that are not multiples
    // of the launch block size are exercised on purpose.
    ScaleTask task{d_in, d_out, 3.f, d_table};
    std::vector<unsigned char> output(N, 0);
    backend->parallel_invoke("scale_task", N, output, &task, sizeof(task));
    backend->sync();

    backend->device_copy_from(out.data(), d_out, N * sizeof(float));

    bool ok = true;
    for (size_t i = 0; i < N; ++i)
        ok &= std::fabs(out[i] - (3.f * in[i] + gl_a)) < 1e-5f;
    Check("named kernel with task struct", ok);

    // Runtime-indexed global array: item k writes gl_table[k] = k + 100.
    int table[8];
    backend->device_copy_from(table, backend->get_global_ptr("gl_table"), sizeof(table));
    bool tableOK = true;
    for (int i = 0; i < 8; ++i)
        tableOK &= table[i] == (i < int(N) ? int(i) + 100 : 0);
    Check("runtime-indexed global array", tableOK);

    // Writes through an argument pointer must show the same behavior.
    int atable[8];
    backend->device_copy_from(atable, d_table, 8 * sizeof(int));
    bool argOK = true;
    for (int i = 0; i < 8; ++i)
        argOK &= atable[i] == (i < int(N) ? i : 0);
    Check("argument-pointer writes", argOK);

    // Registered-pointer global: the blob holds a pointer into shared
    // memory; device code dereferences it (pbrt's Options pattern).
    ProbeOptions *sharedOptions =
        reinterpret_cast<ProbeOptions *>(backend->shared_malloc(sizeof(ProbeOptions)));
    sharedOptions->useGPU = backend->name() != "CPU";
    sharedOptions->scale = 0.5f;
    backend->global_copy_to("prt_probe_options", &sharedOptions, sizeof(sharedOptions));

    int *probeResult = reinterpret_cast<int *>(backend->device_malloc(8 * sizeof(int)));
    OptionsProbeTask probeTask{probeResult};
    std::vector<unsigned char> probeOut(8, 0);
    backend->parallel_invoke("options_probe", 8, probeOut, &probeTask, sizeof(probeTask));
    backend->sync();

    int seen[8];
    backend->device_copy_from(seen, probeResult, 8 * sizeof(int));
    bool optsOK = true;
    for (int i = 0; i < 8; ++i)
        optsOK &= seen[i] == sharedOptions->useGPU;
    Check("registered-pointer global read", optsOK);

    backend->device_free(probeResult);
    backend->device_free(d_in);
    backend->device_free(d_out);
    backend->device_free(d_table);
    return ok && tableOK && argOK && optsOK;
}

int main(int argc, char **argv) {
    bool wantSYCL = argc > 1 && std::strcmp(argv[1], "--sycl") == 0;

    prt::kernelapi_init({{"gl_a", sizeof(float)},
                         {"gl_table", sizeof(gl_table)},
                         {"prt_probe_options", sizeof(void *)}});

    for (prt::Backend *backend : prt::available_backends()) {
        bool isSYCL = backend->name() != "CPU";
        if (isSYCL != wantSYCL)
            continue;

        printf("backend %s (%s)\n", backend->name().c_str(), backend->device_name().c_str());
        backend->shutdown();
        backend->init();

        for (size_t N : {size_t(1), size_t(7), size_t(8), size_t(9), size_t(19), size_t(64)})
            TestNamedKernel(backend, N);

        backend->shutdown();
    }

    printf(failures ? "FAILURES: %d\n" : "all passed (%d failures)\n", failures);
    return failures ? 1 : 0;
}
