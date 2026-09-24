#include <portableRT/kernel/device_common.hpp>





#define PBRT_RESTRICT
#define PBRT_BUILD_PORTABLERT


#include <pbrt/wavefront/integrator.h>
pbrt::WavefrontPathIntegrator* g_integrator;

#include <pbrt/util/spectrum.cpp>
#include <pbrt/util/transform.cpp>
#include <pbrt/film.cpp>
#include <pbrt/wavefront/film.cpp>
#include <pbrt/wavefront/integrator.cpp>


