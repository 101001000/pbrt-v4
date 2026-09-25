#pragma once 


#define PBRT_SAMPLERS(X)                  \
    X(PMJ02BNSampler)                     \
    X(IndependentSampler)                 \
    X(StratifiedSampler)                  \
    X(HaltonSampler)                      \
    X(PaddedSobolSampler)                 \
    X(SobolSampler)                       \
    X(ZSobolSampler)

template <typename T>
inline const char* instantiation_name();

#define DEFINE_INSTANTIATION_NAME(type)             \
    template <>                                     \
    inline const char* instantiation_name<pbrt::type>() {  \
        return #type;                               \
    }

PBRT_SAMPLERS(DEFINE_INSTANTIATION_NAME)

#undef DEFINE_INSTANTIATION_NAME