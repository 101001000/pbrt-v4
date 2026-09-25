// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_UTIL_H
#define PBRT_GPU_UTIL_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/progressreporter.h>

#include <map>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <queue>

#ifdef PBRT_BUILD_PORTABLERT
#include <portableRT/portableRT.hpp>

namespace pbrt {

// The backend selected by GPUInit(); null before GPUInit.
prt::Backend *PRTBackend();

struct EnqueuedArgs {
    void* data;
    size_t size;
};

extern std::queue<EnqueuedArgs> enqueued_args;
extern std::string instance_type;; // Ugly solution to avoid templated kernels.

template <typename... Args>
void enqueue_args(const Args&... args) {
    static_assert((std::is_trivially_copyable_v<Args> && ...));

    const size_t total_size = (sizeof(Args) + ...);
    void* data = std::malloc(total_size);

    size_t offset = 0;

    auto copy_arg = [&](const auto& arg) {
        std::memcpy(static_cast<char*>(data) + offset, &arg, sizeof(arg));
        offset += sizeof(arg);
    };

    (copy_arg(args), ...);

    enqueued_args.push({data, total_size});
}

// Uploads size bytes from src into the kernel-visible global named name,
// which GPUInit has registered.
void PRTCopyToSymbol(const char *name, const void *src, size_t size);

// GPUInit registers the kernel-visible globals with this table; keep it in
// sync with the extern declarations in the kernel translation unit.
#define PBRT_COPY_TO_SYMBOL(name, src, size) \
    pbrt::PRTCopyToSymbol(#name, src, size)

void GPUInit();
void GPUThreadInit();

void GPUWait();

void GPUMemset(void *ptr, int byte, size_t bytes);

template <typename F>
void GPUParallelFor(const char *description, int nItems, F func);

template <typename F>
void GPUParallelFor(const char *description, int nItems, F func) {
    //std::cout << "Invoking parallel for: " << description << std::endl;
#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif
    auto make_kernel_name = [](const char* description) {
        std::string result(description);
        result.erase(
            std::remove(result.begin(), result.end(), ' '),
            result.end()
        );
        result += "PRTKernel";
        if(instance_type != "")
            result += "_" + instance_type;
        return result;
    };
    std::vector<unsigned char> dummy_output(0);

    struct EmptyArgs{};
    EmptyArgs nil;

    EnqueuedArgs args = enqueued_args.empty()
                        ? EnqueuedArgs{&nil, 1}
                        : enqueued_args.front();

    if (!enqueued_args.empty()) {
        enqueued_args.pop();
    }

    PRTBackend()->parallel_invoke(
        make_kernel_name(description),
        nItems,
        dummy_output,
        args.data,
        args.size,
        0
    );
    //std::free(args.data);
    instance_type = "";

    std::cout << "Invoked parallel for: " << description << std::endl;
#ifdef PBRT_DEBUG_BUILD
    PRTBackend()->sync();
        std::cout << "Sync parallel for: " << description << std::endl;
    LOG_VERBOSE("Post-sync %s", description);
#endif
}

}  // namespace pbrt

#else  // !PBRT_BUILD_PORTABLERT

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef NVTX
#ifdef UNICODE
#undef UNICODE
#endif
#include <nvtx3/nvToolsExt.h>

#ifdef RGB
#undef RGB
#endif  // RGB
#endif

#define CUDA_CHECK(EXPR)                                        \
    if (EXPR != cudaSuccess) {                                  \
        cudaError_t error = cudaGetLastError();                 \
        LOG_FATAL("CUDA error: %s", cudaGetErrorString(error)); \
    } else /* eat semicolon */

#define CU_CHECK(EXPR)                                              \
    do {                                                            \
        CUresult result = EXPR;                                     \
        if (result != CUDA_SUCCESS) {                               \
            const char *str;                                        \
            CHECK_EQ(CUDA_SUCCESS, cuGetErrorString(result, &str)); \
            LOG_FATAL("CUDA error: %s", str);                       \
        }                                                           \
    } while (false) /* eat semicolon */

namespace pbrt {

std::pair<cudaEvent_t, cudaEvent_t> GetProfilerEvents(const char *description);

// Uploads size bytes from src into the __constant__ symbol name.
#define PBRT_COPY_TO_SYMBOL(name, src, size) \
    CUDA_CHECK(cudaMemcpyToSymbol(name, src, size))

template <typename F>
inline int GetBlockSize(const char *description, F kernel) {
    // Note: this isn't reentrant, but that's fine for our purposes...
    static std::map<std::type_index, int> kernelBlockSizes;

    std::type_index index = std::type_index(typeid(F));

    auto iter = kernelBlockSizes.find(index);
    if (iter != kernelBlockSizes.end())
        return iter->second;

    int minGridSize, blockSize;
    CUDA_CHECK(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
    kernelBlockSizes[index] = blockSize;
    LOG_VERBOSE("[%s]: block size %d", description, blockSize);

    return blockSize;
}

#ifdef __NVCC__
template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;

    func(tid);
}

// GPU Launch Function Declarations
template <typename F>
void GPUParallelFor(const char *description, int nItems, F func);

template <typename F>
void GPUParallelFor(const char *description, int nItems, F func) {
#ifdef NVTX
    nvtxRangePush(description);
#endif
    auto kernel = &Kernel<F>;

    int blockSize = GetBlockSize(description, kernel);
    std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(description);

#ifdef PBRT_DEBUG_BUILD
    LOG_VERBOSE("Launching %s", description);
#endif
    cudaEventRecord(events.first);
    int gridSize = (nItems + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize>>>(func, nItems);
    cudaEventRecord(events.second);

#ifdef PBRT_DEBUG_BUILD
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_VERBOSE("Post-sync %s", description);
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

#endif  // __NVCC__

// GPU Synchronization Function Declarations
void GPUWait();

void ReportKernelStats();

void GPUInit();
void GPUThreadInit();

void GPUMemset(void *ptr, int byte, size_t bytes);

void GPURegisterThread(const char *name);
void GPUNameStream(cudaStream_t stream, const char *name);

}  // namespace pbrt

#endif  // PBRT_BUILD_PORTABLERT

#endif  // PBRT_GPU_UTIL_H
