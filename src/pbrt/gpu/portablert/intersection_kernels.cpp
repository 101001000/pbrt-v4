#ifdef PRT_KERNEL

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
    prtRay.tmax = Infinity;

    const prt::FullHitReg hit = closest_hit(prtRay);
    if (!hit.valid) {
        pbrt::EnqueueWorkAfterMiss(r, mediumSampleQueue, escapedRayQueue);
        return;
    }

    const pbrt::PortableRTTriangleRecord &record =
        triangleRecords[instanceOffsets[hit.instance_id] + hit.primitive_id];

    // Reconstruct the PBRT interaction from the selected triangle. The BVH
    // traversal is done by PortableRT; this second, local Shape::Intersect is
    // only used to construct PBRT's full SurfaceInteraction and preserves the
    // existing material/medium/light queueing path.
    pstd::optional<pbrt::ShapeIntersection> si =
        record.shape.Intersect(r.ray, hit.t + 1e-4f);
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

#endif