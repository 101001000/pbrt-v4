#pragma once


namespace pbrt {

class PMJ02BNSampler;
class IndependentSampler;
class StratifiedSampler;
class HaltonSampler;
class PaddedSobolSampler;
class SobolSampler;
class ZSobolSampler;

}

#define PBRT_SAMPLERS(X)                  \
    X(HaltonSampler)                      \
    X(PMJ02BNSampler)                     \
    X(IndependentSampler)                 \
    X(StratifiedSampler)                  \
    X(PaddedSobolSampler)                 \
    X(SobolSampler)                       \
    X(ZSobolSampler) 

#define PBRT_EVALUATE_MATERIALS(X)        \
    X(CoatedDiffuseMaterial)              \
    X(DiffuseMaterial)                    \
    X(CoatedConductorMaterial)            \
    X(ConductorMaterial)                  \
    X(DielectricMaterial)                 \
    X(DiffuseTransmissionMaterial)        \
    X(HairMaterial)                       \
    X(MeasuredMaterial)                   \
    X(SubsurfaceMaterial)                 \
    X(ThinDielectricMaterial)

template <typename T>
inline const char *instantiation_name();

template <typename T, typename U>
inline const char *instantiation_name();

#define DEFINE_INSTANTIATION_NAME(type)                   \
    template <>                                           \
    inline const char *instantiation_name<pbrt::type>() { \
        return #type;                                     \
    }

PBRT_SAMPLERS(DEFINE_INSTANTIATION_NAME)

#undef DEFINE_INSTANTIATION_NAME

