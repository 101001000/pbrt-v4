// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#ifndef PBRT_GPU_PORTABLERT_AGGREGATE_H
#define PBRT_GPU_PORTABLERT_AGGREGATE_H

#include <pbrt/pbrt.h>

#include <pbrt/base/shape.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/wavefront/integrator.h>
#include <pbrt/wavefront/workitems.h>

#include <map>
#include <string>
#include <vector>

namespace pbrt {

// Per-triangle information needed after PortableRT returns the closest hit.
// Shape and the referenced PBRT objects are allocated through PBRT's shared
// allocator, so the same records can be read by a PRT kernel.
struct PortableRTTriangleRecord {
    Triangle triangle;
    Material material;
    Light areaLight;
    MediumInterface mediumInterface;
};

class PortableRTAggregate final : public WavefrontAggregate {
  public:
    PortableRTAggregate(
        BasicScene &scene, NamedTextures &textures,
        const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
        const std::map<std::string, Medium> &media,
        const std::map<std::string, pbrt::Material> &namedMaterials,
        const std::vector<pbrt::Material> &materials);

    Bounds3f Bounds() const override { return bounds; }

    void IntersectClosest(int maxRays, const RayQueue *rayQueue,
                          EscapedRayQueue *escapedRayQueue,
                          HitAreaLightQueue *hitAreaLightQueue,
                          MaterialEvalQueue *basicEvalMaterialQueue,
                          MaterialEvalQueue *universalEvalMaterialQueue,
                          MediumSampleQueue *mediumSampleQueue,
                          RayQueue *nextRayQueue) const override;

    void IntersectShadow(int, ShadowRayQueue *, SOA<PixelSampleState> *) const override;
    void IntersectShadowTr(int, ShadowRayQueue *, SOA<PixelSampleState> *) const override;
    void IntersectOneRandom(int, SubsurfaceScatterQueue *) const override;

  private:
    pstd::vector<PortableRTTriangleRecord> triangleRecords;
    pstd::vector<int> instanceOffsets;
    Bounds3f bounds;
};

}  // namespace pbrt

#endif  // PBRT_GPU_PORTABLERT_AGGREGATE_H
