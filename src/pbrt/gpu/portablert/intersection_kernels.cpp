#include <pbrt/pbrt.h>
#include <pbrt/gpu/portablert/aggregate.h>
#include <pbrt/wavefront/intersect.h>
#include <pbrt/shapes.h>
#include <pbrt/util/float.h>

#include <limits>


#ifdef PRT_KERNEL

prt_kernel void dummy(){
    auto hit = prt::closest_hit(prt::Ray());
    auto a = pbrt::allTriangleMeshesGPU;
    auto b = pbrt::allBilinearMeshesGPU;
}

// This kernel is intended to be included by the PortableRT kernel translation
// unit. PortableRT rewrites closest_hit() to the selected BVH implementation.
prt_kernel void PortableRTintersectclosestPRTKernel(
    const pbrt::RayQueue *rayQueue, pbrt::EscapedRayQueue *escapedRayQueue,
    pbrt::HitAreaLightQueue *hitAreaLightQueue,
    pbrt::MaterialEvalQueue *basicEvalMaterialQueue,
    pbrt::MaterialEvalQueue *universalEvalMaterialQueue,
    pbrt::MediumSampleQueue *mediumSampleQueue, pbrt::RayQueue *nextRayQueue,
    const pbrt::PortableRTTriangleRecord *triangleRecords,
    const int *instanceOffsets) {
    const int index = global_idx;
    if (index >= rayQueue->Size())
        return;

    const pbrt::RayWorkItem r = (*rayQueue)[index];

    prt::Ray prtRay;
    prtRay.origin[0] = r.ray.o.x;
    prtRay.origin[1] = r.ray.o.y;
    prtRay.origin[2] = r.ray.o.z;
    prtRay.direction[0] = r.ray.d.x;
    prtRay.direction[1] = r.ray.d.y;
    prtRay.direction[2] = r.ray.d.z;
    prtRay.tmin = 0.0f;
    prtRay.tmax = std::numeric_limits<float>::infinity();

    const prt::FullHitReg hit = closest_hit(prtRay);
    if (!hit.valid) {
        pbrt::EnqueueWorkAfterMiss(r, mediumSampleQueue, escapedRayQueue);
        return;
    }

    const pbrt::PortableRTTriangleRecord &record =
        triangleRecords[instanceOffsets[hit.instance_id] + hit.primitive_id];

    // Reconstruct the PBRT interaction from the selected triangle. The BVH
    // traversal is done by PortableRT; this second, local Triangle::Intersect is
    // only used to construct PBRT's full SurfaceInteraction and preserves the
    // existing material/medium/light queueing path.
    pstd::optional<pbrt::ShapeIntersection> si =
        record.triangle.Intersect(r.ray, hit.t + 1e-4f);
    if (!si) {
        pbrt::EnqueueWorkAfterMiss(r, mediumSampleQueue, escapedRayQueue);
        return;
    }

    si->intr.material = record.material;
    si->intr.areaLight = record.areaLight;
    si->intr.mediumInterface = &record.mediumInterface;

    pbrt::EnqueueWorkAfterIntersection(
        r, r.ray.medium, si->tHit, si->intr, mediumSampleQueue, nextRayQueue,
        hitAreaLightQueue, basicEvalMaterialQueue, universalEvalMaterialQueue);
}

// This kernel is intended to be included by the PortableRT kernel translation
// unit. As with PortableRTintersectclosestPRTKernel, closest_hit() is
// rewritten to the selected BVH implementation.
prt_kernel void PortableRTintersectshadowPRTKernel(
    pbrt::ShadowRayQueue *shadowRayQueue,
    pbrt::SOA<pbrt::PixelSampleState> *pixelSampleState) {
    const int index = global_idx;
    if (index >= shadowRayQueue->Size())
        return;

    const pbrt::ShadowRayWorkItem w = (*shadowRayQueue)[index];

    prt::Ray prtRay;
    prtRay.origin[0] = w.ray.o.x;
    prtRay.origin[1] = w.ray.o.y;
    prtRay.origin[2] = w.ray.o.z;
    prtRay.direction[0] = w.ray.d.x;
    prtRay.direction[1] = w.ray.d.y;
    prtRay.direction[2] = w.ray.d.z;
    prtRay.tmin = 0.0f;
    prtRay.tmax = w.tMax;

    const prt::FullHitReg hit = closest_hit(prtRay);
    pbrt::RecordShadowRayResult(w, pixelSampleState, hit.valid);
}

#endif
