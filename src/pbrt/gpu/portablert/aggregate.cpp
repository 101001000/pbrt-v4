// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include <pbrt/gpu/portablert/aggregate.h>

#include <pbrt/gpu/memory.h>
#include <pbrt/gpu/util.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/log.h>
#include <pbrt/util/mesh.h>
#include <pbrt/wavefront/intersect.h>

#include <portableRT/portableRT.hpp>
#include <type_traits>
#ifdef PRT_KERNEL
#include <portableRT/kernel/device_common.hpp>
#endif

namespace pbrt {


namespace {

// Must match the argument layout emitted by the PortableRT kernel rewriter.
// Keep this package local to the host invocation: using enqueued_args here can
// accidentally consume an argument packet belonging to another kernel.
struct PortableRTIntersectClosestArgs {
    const RayQueue *rayQueue;
    EscapedRayQueue *escapedRayQueue;
    HitAreaLightQueue *hitAreaLightQueue;
    MaterialEvalQueue *basicEvalMaterialQueue;
    MaterialEvalQueue *universalEvalMaterialQueue;
    MediumSampleQueue *mediumSampleQueue;
    RayQueue *nextRayQueue;
    const PortableRTTriangleRecord *triangleRecords;
    const int *instanceOffsets;
};

static_assert(std::is_trivially_copyable_v<PortableRTIntersectClosestArgs>);
static_assert(sizeof(PortableRTIntersectClosestArgs) == 9 * sizeof(void *));

// Must match the argument layout emitted by the PortableRT kernel rewriter.
struct PortableRTIntersectShadowArgs {
    ShadowRayQueue *shadowRayQueue;
    SOA<PixelSampleState> *pixelSampleState;
};

static_assert(std::is_trivially_copyable_v<PortableRTIntersectShadowArgs>);
static_assert(sizeof(PortableRTIntersectShadowArgs) == 2 * sizeof(void *));

static bool IsTriangleShape(const ShapeSceneEntity &shape) {
    return shape.name == "trianglemesh" || shape.name == "plymesh";
}

static Medium LookupMedium(const std::map<std::string, Medium> &media,
                           const std::string &name, const FileLoc *loc) {
    if (name.empty())
        return nullptr;
    auto it = media.find(name);
    if (it == media.end())
        ErrorExit(loc, "%s: medium not defined", name);
    return it->second;
}

}  // namespace

PortableRTAggregate::PortableRTAggregate(
    BasicScene &scene, NamedTextures &textures,
    const std::map<int, pstd::vector<Light> *> &shapeIndexToAreaLights,
    const std::map<std::string, Medium> &media,
    const std::map<std::string, pbrt::Material> &namedMaterials,
    const std::vector<pbrt::Material> &materials)
    : triangleRecords(pstd::pmr::polymorphic_allocator<PortableRTTriangleRecord>(
          &CUDATrackedMemoryResource::singleton)),
      instanceOffsets(pstd::pmr::polymorphic_allocator<int>(
          &CUDATrackedMemoryResource::singleton)) {
    Allocator alloc(&CUDATrackedMemoryResource::singleton);
    std::vector<prt::BLASInstance> blases;

    for (size_t shapeIndex = 0; shapeIndex < scene.shapes.size(); ++shapeIndex) {
        const ShapeSceneEntity &sceneShape = scene.shapes[shapeIndex];
        if (!IsTriangleShape(sceneShape))
            continue;

        TriangleMesh *mesh = nullptr;
        if (sceneShape.name == "trianglemesh") {
            mesh = Triangle::CreateMesh(sceneShape.renderFromObject,
                                        sceneShape.reverseOrientation,
                                        sceneShape.parameters, &sceneShape.loc, alloc);
        } else {
            const std::string filename = ResolveFilename(
                sceneShape.parameters.GetOneString("filename", ""));
            TriQuadMesh ply = TriQuadMesh::ReadPLY(filename);
            ply.ConvertToOnlyTriangles();
            mesh = alloc.new_object<TriangleMesh>(
                *sceneShape.renderFromObject, sceneShape.reverseOrientation,
                ply.triIndices, ply.p, std::vector<Vector3f>(), ply.n, ply.uv,
                ply.faceIndices, alloc);
        }

        if (!mesh || mesh->nTriangles == 0)
            continue;

        pstd::vector<Shape> shapes = Triangle::CreateTriangles(mesh, alloc);
        if (shapes.empty())
            continue;

        Material material = nullptr;
        if (!sceneShape.materialName.empty()) {
            auto materialIt = namedMaterials.find(sceneShape.materialName);
            if (materialIt == namedMaterials.end())
                ErrorExit(&sceneShape.loc, "%s: no named material defined.",
                          sceneShape.materialName);
            material = materialIt->second;
        } else {
            CHECK_LT(sceneShape.materialIndex, materials.size());
            material = materials[sceneShape.materialIndex];
        }

        MediumInterface mediumInterface(
            LookupMedium(media, sceneShape.insideMedium, &sceneShape.loc),
            LookupMedium(media, sceneShape.outsideMedium, &sceneShape.loc));

        const auto areaLightIt = shapeIndexToAreaLights.find(int(shapeIndex));
        const pstd::vector<Light> *areaLights =
            areaLightIt == shapeIndexToAreaLights.end() ? nullptr : areaLightIt->second;

        instanceOffsets.push_back(int(triangleRecords.size()));
        prt::Tris tris;
        tris.reserve(mesh->nTriangles);

        for (int triIndex = 0; triIndex < mesh->nTriangles; ++triIndex) {
            const int *indices = &mesh->vertexIndices[3 * triIndex];
            const Point3f &p0 = mesh->p[indices[0]];
            const Point3f &p1 = mesh->p[indices[1]];
            const Point3f &p2 = mesh->p[indices[2]];

            tris.push_back({p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z});
            bounds = Union(bounds, p0);
            bounds = Union(bounds, p1);
            bounds = Union(bounds, p2);

            Light areaLight = nullptr;
            if (areaLights && triIndex < areaLights->size())
                areaLight = (*areaLights)[triIndex];

            triangleRecords.push_back(
                PortableRTTriangleRecord{*shapes[triIndex].CastOrNullptr<Triangle>(),
                                         material, areaLight, mediumInterface});
        }

        // One BLAS per PBRT mesh preserves instance_id -> mesh/shape mapping.
        blases.emplace_back(prt::BLAS(tris));
    }

    if (blases.empty()) {
        LOG_ERROR("PortableRTAggregate: scene contains no triangle meshes");
        return;
    }

    if (PRTBackend() == nullptr)
        LOG_FATAL("PortableRTAggregate constructed before GPUInit");

    PRTBackend()->set_tris(prt::TLAS(blases));
}

void PortableRTAggregate::IntersectClosest(
    int maxRays, const RayQueue *rayQueue, EscapedRayQueue *escapedRayQueue,
    HitAreaLightQueue *hitAreaLightQueue, MaterialEvalQueue *basicEvalMaterialQueue,
    MaterialEvalQueue *universalEvalMaterialQueue, MediumSampleQueue *mediumSampleQueue,
    RayQueue *nextRayQueue) const {

    const PortableRTIntersectClosestArgs args{
        rayQueue,
        escapedRayQueue,
        hitAreaLightQueue,
        basicEvalMaterialQueue,
        universalEvalMaterialQueue,
        mediumSampleQueue,
        nextRayQueue,
        triangleRecords.data(),
        instanceOffsets.data()};

    std::vector<unsigned char> output;
    PRTParallelInvoke(
        "PortableRTintersectclosestPRTKernel", maxRays, output, &args, sizeof(args));
}

void PortableRTAggregate::IntersectShadow(
    int maxRays, ShadowRayQueue *shadowRayQueue,
    SOA<PixelSampleState> *pixelSampleState) const {
    const PortableRTIntersectShadowArgs args{shadowRayQueue, pixelSampleState};

    std::vector<unsigned char> output;
    PRTParallelInvoke(
        "PortableRTintersectshadowPRTKernel", maxRays, output, &args, sizeof(args));
}

void PortableRTAggregate::IntersectShadowTr(int, ShadowRayQueue *,
                                            SOA<PixelSampleState> *) const {
    LOG_FATAL("PortableRTAggregate::IntersectShadowTr is not implemented");
}

void PortableRTAggregate::IntersectOneRandom(int, SubsurfaceScatterQueue *) const {
    LOG_FATAL("PortableRTAggregate::IntersectOneRandom is not implemented");
}

}  // namespace pbrt
