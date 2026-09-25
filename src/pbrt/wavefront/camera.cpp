// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/cameras.h>
#include <pbrt/options.h>
#include <pbrt/samplers.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/wavefront/integrator.h>

#define PBRT_SAMPLERS(X)                  \
    X(PMJ02BNSampler)                     \
    X(IndependentSampler)                 \
    X(StratifiedSampler)                  \
    X(HaltonSampler)                      \
    X(PaddedSobolSampler)                 \
    X(SobolSampler)                       \
    X(ZSobolSampler)

template <typename T>
const char* instantiation_name();

#define DEFINE_INSTANTIATION_NAME(type)             \
    template <>                                     \
    const char* instantiation_name<pbrt::type>() {  \
        return #type;                               \
    }

PBRT_SAMPLERS(DEFINE_INSTANTIATION_NAME)

#undef DEFINE_INSTANTIATION_NAME

#ifdef PRT_KERNEL

#define DEFINE_GENERATE_CAMERA_RAYS_KERNEL(sampler_type)                           \
prt_kernel void GeneratecameraraysPRTKernel_##sampler_type(                        \
    int y0, pbrt::Transform moving_from_camera, int sample_index) {                \
                                                                                   \
    int pixel_index = global_idx;                                                  \
    pbrt::RayQueue* ray_queue = g_integrator->CurrentRayQueue(0);                  \
                                                                                   \
    pbrt::Bounds2i pixel_bounds = g_integrator->film.PixelBounds();                \
    int x_resolution = pixel_bounds.pMax.x - pixel_bounds.pMin.x;                  \
    pbrt::Point2i p_pixel(                                                         \
        pixel_bounds.pMin.x + pixel_index % x_resolution,                         \
        y0 + pixel_index / x_resolution);                                         \
                                                                                   \
    g_integrator->pixelSampleState.pPixel[pixel_index] = p_pixel;                 \
                                                                                   \
    if (!pbrt::InsideExclusive(p_pixel, pixel_bounds))                             \
        return;                                                                    \
                                                                                   \
    pbrt::sampler_type pixel_sampler =                                             \
        *g_integrator->sampler.Cast<pbrt::sampler_type>();                         \
                                                                                   \
    pixel_sampler.StartPixelSample(p_pixel, sample_index, 0);                      \
                                                                                   \
    pbrt::Float lu = pixel_sampler.Get1D();                                        \
    if (pbrt::GetOptions().disableWavelengthJitter)                                \
        lu = 0.5f;                                                                 \
                                                                                   \
    pbrt::SampledWavelengths lambda =                                              \
        g_integrator->film.SampleWavelengths(lu);                                  \
                                                                                   \
    pbrt::CameraSample camera_sample =                                             \
        pbrt::GetCameraSample(                                                     \
            pixel_sampler, p_pixel, g_integrator->filter);                        \
                                                                                   \
    pstd::optional<pbrt::CameraRay> camera_ray =                                   \
        g_integrator->camera.GenerateRay(camera_sample, lambda);                  \
                                                                                   \
    if (camera_ray)                                                                \
        camera_ray->ray = moving_from_camera(camera_ray->ray);                    \
                                                                                   \
    g_integrator->pixelSampleState.L[pixel_index] =                                \
        pbrt::SampledSpectrum(0.f);                                                \
    g_integrator->pixelSampleState.lambda[pixel_index] = lambda;                  \
    g_integrator->pixelSampleState.filterWeight[pixel_index] =                    \
        camera_sample.filterWeight;                                                \
                                                                                   \
    if (g_integrator->initializeVisibleSurface)                                   \
        g_integrator->pixelSampleState.visibleSurface[pixel_index] =              \
            pbrt::VisibleSurface();                                                \
                                                                                   \
    if (camera_ray) {                                                             \
        ray_queue->PushCameraRay(camera_ray->ray, lambda, pixel_index);           \
        g_integrator->pixelSampleState.cameraRayWeight[pixel_index] =             \
            camera_ray->weight;                                                   \
    } else {                                                                       \
        g_integrator->pixelSampleState.cameraRayWeight[pixel_index] =             \
            pbrt::SampledSpectrum(0);                                              \
    }                                                                              \
}

PBRT_SAMPLERS(DEFINE_GENERATE_CAMERA_RAYS_KERNEL)

#endif

namespace pbrt {

// WavefrontPathIntegrator Camera Ray Methods
void WavefrontPathIntegrator::GenerateCameraRays(int y0, Transform movingFromCamera,
                                                 int sampleIndex) {
    // Define _generateRays_ lambda function
    auto generateRays = [=](auto sampler) {
        using ConcreteSampler = std::remove_reference_t<decltype(*sampler)>;
        if constexpr (!std::is_same_v<ConcreteSampler, MLTSampler> &&
                      !std::is_same_v<ConcreteSampler, DebugMLTSampler>)
            GenerateCameraRays<ConcreteSampler>(y0, movingFromCamera, sampleIndex);
    };

    sampler.DispatchCPU(generateRays);
}

template <typename ConcreteSampler>
void WavefrontPathIntegrator::GenerateCameraRays(int y0, Transform movingFromCamera,
                                                 int sampleIndex) {
    RayQueue *rayQueue = CurrentRayQueue(0);
    enqueue_args(y0, movingFromCamera, sampleIndex);
    instance_type = instantiation_name<ConcreteSampler>();
    ParallelFor(
        "Generate camera rays", maxQueueSize, PBRT_CPU_GPU_LAMBDA(int pixelIndex) {
            // Enqueue camera ray and set pixel state for sample
            // Compute pixel coordinates for _pixelIndex_
            Bounds2i pixelBounds = film.PixelBounds();
            int xResolution = pixelBounds.pMax.x - pixelBounds.pMin.x;
            Point2i pPixel(pixelBounds.pMin.x + pixelIndex % xResolution,
                           y0 + pixelIndex / xResolution);
            pixelSampleState.pPixel[pixelIndex] = pPixel;

            // Test pixel coordinates against pixel bounds
            if (!InsideExclusive(pPixel, pixelBounds))
                return;

            // Initialize _Sampler_ for current pixel and sample
            ConcreteSampler pixelSampler = *sampler.Cast<ConcreteSampler>();
            pixelSampler.StartPixelSample(pPixel, sampleIndex, 0);

            // Sample wavelengths for ray path
            Float lu = pixelSampler.Get1D();
            if (GetOptions().disableWavelengthJitter)
                lu = 0.5f;
            SampledWavelengths lambda = film.SampleWavelengths(lu);

            // Generate _CameraSample_ and corresponding ray
            CameraSample cameraSample = GetCameraSample(pixelSampler, pPixel, filter);
            pstd::optional<CameraRay> cameraRay =
                camera.GenerateRay(cameraSample, lambda);
            if (cameraRay)
                cameraRay->ray = movingFromCamera(cameraRay->ray);

            // Initialize remainder of _PixelSampleState_ for ray
            pixelSampleState.L[pixelIndex] = SampledSpectrum(0.f);
            pixelSampleState.lambda[pixelIndex] = lambda;
            pixelSampleState.filterWeight[pixelIndex] = cameraSample.filterWeight;
            if (initializeVisibleSurface)
                pixelSampleState.visibleSurface[pixelIndex] = VisibleSurface();

            // Enqueue camera ray for intersection tests
            if (cameraRay) {
                rayQueue->PushCameraRay(cameraRay->ray, lambda, pixelIndex);
                pixelSampleState.cameraRayWeight[pixelIndex] = cameraRay->weight;
            } else
                pixelSampleState.cameraRayWeight[pixelIndex] = SampledSpectrum(0);
        });
}

}  // namespace pbrt
