#pragma once

#if PRT_KERNEL_CPU

template <typename T>
inline T atomic_load_relaxed(const T *ptr) {
    T value;
    __atomic_load(ptr, &value, __ATOMIC_RELAXED);
    return value;
}

template <typename T>
inline void atomic_store_relaxed(T *ptr, T value) {
    __atomic_store(ptr, &value, __ATOMIC_RELAXED);
}

template <typename T>
inline T atomic_fetch_add_relaxed(T *ptr, T value) {
    T old = atomic_load_relaxed(ptr);

    while (true) {
        T desired = old + value;
        if (__atomic_compare_exchange(ptr, &old, &desired, false,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return old;
    }
}

#elif PRT_KERNEL_CUDA

#include <cuda/atomic>

template <typename T>
__host__ __device__ inline T atomic_load_relaxed(const T *ptr) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(
        *const_cast<T *>(ptr));
    return ref.load(cuda::memory_order_relaxed);
}

template <typename T>
__host__ __device__ inline void atomic_store_relaxed(T *ptr, T value) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    ref.store(value, cuda::memory_order_relaxed);
}

template <typename T>
__host__ __device__ inline T atomic_fetch_add_relaxed(T *ptr, T value) {
    cuda::atomic_ref<T, cuda::thread_scope_device> ref(*ptr);
    return ref.fetch_add(value, cuda::memory_order_relaxed);
}

#elif PRT_KERNEL_HIP

template <typename T>
__host__ __device__ inline T atomic_load_relaxed(const T *ptr) {
    T value;
    __scoped_atomic_load(
        ptr,
        &value,
        __ATOMIC_RELAXED,
        __MEMORY_SCOPE_DEVICE);
    return value;
}

template <typename T>
__host__ __device__ inline void atomic_store_relaxed(T *ptr, T value) {
    __scoped_atomic_store(
        ptr,
        &value,
        __ATOMIC_RELAXED,
        __MEMORY_SCOPE_DEVICE);
}

template <typename T>
__host__ __device__ inline T atomic_fetch_add_relaxed(T *ptr, T value) {
    return __scoped_atomic_fetch_add(
        ptr,
        value,
        __ATOMIC_RELAXED,
        __MEMORY_SCOPE_DEVICE);
}

#elif PRT_KERNEL_SYCL

#include <sycl/sycl.hpp>

template <typename T>
inline T atomic_load_relaxed(const T *ptr) {
    using AtomicRef = sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>;

    return AtomicRef(*const_cast<T *>(ptr)).load();
}

template <typename T>
inline void atomic_store_relaxed(T *ptr, T value) {
    using AtomicRef = sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>;

    AtomicRef(*ptr).store(value);
}

template <typename T>
inline T atomic_fetch_add_relaxed(T *ptr, T value) {
    using AtomicRef = sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>;

    return AtomicRef(*ptr).fetch_add(value);
}

#endif