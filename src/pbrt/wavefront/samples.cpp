// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/samplers.h>
#include <pbrt/wavefront/integrator.h>

#include <type_traits>

#ifdef interface
#undef interface
#endif  // interface

#ifdef PBRT_BUILD_PORTABLERT
#include <pbrt/portablert/samplers.h>
#endif

#ifdef PRT_KERNEL

#define DEFINE_GENERATE_RAY_SAMPLES_KERNEL(sampler_type)                           \
prt_kernel void Generateraysamples_##sampler_type##PRTKernel( \
    pbrt::RayQueue *ray_queue, int sample_index) {                                       \
                                                                                   \
    int index = global_idx;                                                        \
    if (index >= ray_queue->Size())                                                \
        return;                                                                    \
                                                                                   \
    const pbrt::RayWorkItem w = (*ray_queue)[index];                               \
                                                                                   \
    /* Find first sample dimension */                                              \
    int dimension = 6 + 7 * w.depth;                                              \
    if (g_integrator->haveSubsurface)                                              \
        dimension += 3 * w.depth;                                                 \
                                                                                   \
    /* Initialize sampler for pixel, sample index, and dimension */                \
    pbrt::sampler_type pixel_sampler =                                             \
        *g_integrator->sampler.Cast<pbrt::sampler_type>();                         \
                                                                                   \
    pbrt::Point2i p_pixel =                                                        \
        g_integrator->pixelSampleState.pPixel[w.pixelIndex];                       \
                                                                                   \
    pixel_sampler.StartPixelSample(p_pixel, sample_index, dimension);              \
                                                                                   \
    /* Initialize RaySamples structure with sample values */                       \
    pbrt::RaySamples rs;                                                           \
    rs.direct.uc = pixel_sampler.Get1D();                                          \
    rs.direct.u = pixel_sampler.Get2D();                                           \
                                                                                   \
    rs.indirect.uc = pixel_sampler.Get1D();                                        \
    rs.indirect.u = pixel_sampler.Get2D();                                         \
    rs.indirect.rr = pixel_sampler.Get1D();                                        \
                                                                                   \
    rs.haveSubsurface = g_integrator->haveSubsurface;                              \
    if (g_integrator->haveSubsurface) {                                            \
        rs.subsurface.uc = pixel_sampler.Get1D();                                  \
        rs.subsurface.u = pixel_sampler.Get2D();                                   \
    }                                                                              \
                                                                                   \
    /* Store RaySamples in pixel sample state */                                   \
    g_integrator->pixelSampleState.samples[w.pixelIndex] = rs;                     \
}

#define PBRT_SAMPLERS_THIN(X)                  \
    X(HaltonSampler)                      
//PBRT_SAMPLERS(DEFINE_GENERATE_CAMERA_RAYS_KERNEL)
PBRT_SAMPLERS_THIN(DEFINE_GENERATE_RAY_SAMPLES_KERNEL)

#endif


namespace pbrt {

// WavefrontPathIntegrator Sampler Methods
void WavefrontPathIntegrator::GenerateRaySamples(int wavefrontDepth, int sampleIndex) {
    auto generateSamples = [=](auto sampler) {
        using ConcreteSampler = std::remove_reference_t<decltype(*sampler)>;
        if constexpr (!std::is_same_v<ConcreteSampler, MLTSampler> &&
                      !std::is_same_v<ConcreteSampler, DebugMLTSampler>)
            GenerateRaySamples<ConcreteSampler>(wavefrontDepth, sampleIndex);
    };
    sampler.DispatchCPU(generateSamples);
}

template <typename ConcreteSampler>
void WavefrontPathIntegrator::GenerateRaySamples(int wavefrontDepth, int sampleIndex) {
    // Generate description string _desc_ for ray sample generation
    std::string desc = std::string("Generate ray samples - ") + ConcreteSampler::Name();

    RayQueue *rayQueue = CurrentRayQueue(wavefrontDepth);
    enqueue_args(rayQueue, sampleIndex);
   // instance_type = instantiation_name<ConcreteSampler>();
    ForAllQueued(
        desc.c_str(), rayQueue, maxQueueSize, PBRT_CPU_GPU_LAMBDA(const RayWorkItem w) {
            // Generate samples for ray segment at current sample index
            // Find first sample dimension
            int dimension = 6 + 7 * w.depth;
            if (haveSubsurface)
                dimension += 3 * w.depth;

            // Initialize _Sampler_ for pixel, sample index, and dimension
            ConcreteSampler pixelSampler = *sampler.Cast<ConcreteSampler>();
            Point2i pPixel = pixelSampleState.pPixel[w.pixelIndex];
            pixelSampler.StartPixelSample(pPixel, sampleIndex, dimension);

            // Initialize _RaySamples_ structure with sample values
            RaySamples rs;
            rs.direct.uc = pixelSampler.Get1D();
            rs.direct.u = pixelSampler.Get2D();
            // Initialize remaining samples in _rs_
            rs.indirect.uc = pixelSampler.Get1D();
            rs.indirect.u = pixelSampler.Get2D();
            rs.indirect.rr = pixelSampler.Get1D();
            // Possibly initialize subsurface subsurface samples in _rs_
            rs.haveSubsurface = haveSubsurface;
            if (haveSubsurface) {
                rs.subsurface.uc = pixelSampler.Get1D();
                rs.subsurface.u = pixelSampler.Get2D();
            }

            // Store _RaySamples_ in pixel sample state
            pixelSampleState.samples[w.pixelIndex] = rs;
        });
}

}  // namespace pbrt
