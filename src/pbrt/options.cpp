// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/options.h>

#if defined(PBRT_BUILD_GPU_RENDERER) || defined(PBRT_BUILD_PORTABLERT)
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#if defined(PBRT_BUILD_PORTABLERT)
#include <portableRT/portableRT.hpp>
#endif
#include <pbrt/util/print.h>

namespace pbrt {

PBRTOptions *Options;

#if defined(PBRT_BUILD_GPU_RENDERER)
__constant__ BasicPBRTOptions OptionsGPU;

void CopyOptionsToGPU() {
    PBRT_COPY_TO_SYMBOL(OptionsGPU, Options, sizeof(OptionsGPU));
}
#elif defined(PBRT_BUILD_PORTABLERT)
void CopyOptionsToGPU() {
    // Device code dereferences pbrt::Options through the registered global
    // pointer, so the options must live in shared memory. construction into
    // the raw shared allocation (assignment would inspect an
    // uninitialized std::string).
    PBRTOptions *shared =
        reinterpret_cast<PBRTOptions *>(PRTBackend()->shared_malloc(sizeof(PBRTOptions)));
    new (shared) PBRTOptions(*Options);
    PBRT_COPY_TO_SYMBOL(Options, &shared, sizeof(shared));
}
#endif

std::string ToString(const RenderingCoordinateSystem &r) {
    if (r == RenderingCoordinateSystem::Camera)
        return "RenderingCoordinateSystem::Camera";
    else if (r == RenderingCoordinateSystem::CameraWorld)
        return "RenderingCoordinateSystem::CameraWorld";
    else {
        CHECK(r == RenderingCoordinateSystem::World);
        return "RenderingCoordinateSystem::World";
    }
}

std::string PBRTOptions::ToString() const {
    return StringPrintf(
        "[ PBRTOptions seed: %s quiet: %s disablePixelJitter: %s "
        "disableWavelengthJitter: %s disableTextureFiltering: %s disableImageTextures: %s "
        "forceDiffuse: %s useGPU: %s wavefront: %s interactive: %s fullscreen %s "
        "renderingSpace: %s nThreads: %s logLevel: %s logFile: %s logUtilization: %s "
        "writePartialImages: %s recordPixelStatistics: %s "
        "printStatistics: %s pixelSamples: %s gpuDevice: %s quickRender: %s upgrade: %s "
        "imageFile: %s mseReferenceImage: %s mseReferenceOutput: %s debugStart: %s "
        "displayServer: %s cropWindow: %s pixelBounds: %s pixelMaterial: %s "
        "displacementEdgeScale: %f ]",
        seed, quiet, disablePixelJitter, disableWavelengthJitter, disableTextureFiltering,
        disableImageTextures, forceDiffuse, useGPU, wavefront, interactive, fullscreen,
        renderingSpace, nThreads, logLevel, logFile, logUtilization, writePartialImages,
        recordPixelStatistics, printStatistics, pixelSamples, gpuDevice, quickRender, upgrade,
        imageFile, mseReferenceImage, mseReferenceOutput, debugStart, displayServer, cropWindow,
        pixelBounds, pixelMaterial, displacementEdgeScale);
}

}  // namespace pbrt
