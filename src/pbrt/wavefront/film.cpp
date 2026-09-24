// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/film.h>
#include <pbrt/wavefront/integrator.h>

pbrt::WavefrontPathIntegrator* g_integrator;

#ifdef PRT_KERNEL

prt_kernel void dummy(){
    auto hit = prt::closest_hit(prt::Ray());
}

prt_kernel void UpdatefilmPRTKernel(){

    int pixelIndex = global_idx;

    // Check pixel against film bounds
    pbrt::Point2i pPixel = g_integrator->pixelSampleState.pPixel[pixelIndex];
    if (!InsideExclusive(pPixel, g_integrator->film.PixelBounds()))
        return;

    // Compute final weighted radiance value
    pbrt::SampledSpectrum Lw = pbrt::SampledSpectrum(g_integrator->pixelSampleState.L[pixelIndex]) *
                         g_integrator->pixelSampleState.cameraRayWeight[pixelIndex];

    //PBRT_DBG("Adding Lw %f %f %f %f at pixel (%d, %d)\n", Lw[0], Lw[1], Lw[2],
    //         Lw[3], pPixel.x, pPixel.y);
    // Provide sample radiance value to film
    pbrt::SampledWavelengths lambda = g_integrator->pixelSampleState.lambda[pixelIndex];
    pbrt::Float filterWeight = g_integrator->pixelSampleState.filterWeight[pixelIndex];
    if (g_integrator->initializeVisibleSurface) {
        // Call _Film::AddSample()_ with _VisibleSurface_ for pixel sample
        pbrt::VisibleSurface visibleSurface =
            g_integrator->pixelSampleState.visibleSurface[pixelIndex];
        g_integrator->film.AddSample(pPixel, Lw, lambda, &visibleSurface, filterWeight);
    } else
        g_integrator->film.AddSample(pPixel, Lw, lambda, nullptr, filterWeight);
}
#else

namespace pbrt {

// WavefrontPathIntegrator Film Methods
void WavefrontPathIntegrator::UpdateFilm() {
    ParallelFor(
        "Update film", maxQueueSize, PBRT_CPU_GPU_LAMBDA(int pixelIndex) {
            // Check pixel against film bounds
            Point2i pPixel = pixelSampleState.pPixel[pixelIndex];
            if (!InsideExclusive(pPixel, film.PixelBounds()))
                return;

            // Compute final weighted radiance value
            SampledSpectrum Lw = SampledSpectrum(pixelSampleState.L[pixelIndex]) *
                                 pixelSampleState.cameraRayWeight[pixelIndex];

            PBRT_DBG("Adding Lw %f %f %f %f at pixel (%d, %d)\n", Lw[0], Lw[1], Lw[2],
                     Lw[3], pPixel.x, pPixel.y);
            // Provide sample radiance value to film
            SampledWavelengths lambda = pixelSampleState.lambda[pixelIndex];
            Float filterWeight = pixelSampleState.filterWeight[pixelIndex];
            if (initializeVisibleSurface) {
                // Call _Film::AddSample()_ with _VisibleSurface_ for pixel sample
                VisibleSurface visibleSurface =
                    pixelSampleState.visibleSurface[pixelIndex];
                film.AddSample(pPixel, Lw, lambda, &visibleSurface, filterWeight);

            } else
                film.AddSample(pPixel, Lw, lambda, nullptr, filterWeight);
        });
}

}  // namespace pbrt
#endif