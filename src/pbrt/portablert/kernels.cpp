#include <gtest/gtest.h> // Required so float.h don't mix infinity

#include <portableRT/kernel/device_common.hpp>

#define PBRT_RESTRICT
#define PBRT_BUILD_PORTABLERT


#include <pbrt/wavefront/integrator.h>
pbrt::WavefrontPathIntegrator* g_integrator;

#include <pbrt/util/spectrum.cpp>
#include <pbrt/util/transform.cpp>
#include <pbrt/film.cpp>
#include <pbrt/wavefront/film.cpp>

#include <pbrt/util/sampling.cpp>
#include <pbrt/shapes.cpp>
#include <pbrt/lightsamplers.cpp>
#include <pbrt/util/colorspace.cpp>
#include <pbrt/lights.cpp>

#include <pbrt/wavefront/integrator.cpp>
#include <pbrt/util/bluenoise.cpp>
#include <pbrt/util/pmj02tables.cpp>
#include <pbrt/util/sobolmatrices.cpp>
#include <pbrt/util/primes.cpp>
#include <pbrt/util/math.cpp>
#include <pbrt/util/color.cpp>
#include <pbrt/cameras.cpp>
#include <pbrt/wavefront/camera.cpp>

#include <pbrt/wavefront/samples.cpp>

#include <pbrt/util/scattering.cpp>
#include <pbrt/util/noise.cpp>
#include <pbrt/bxdfs.cpp>
#include <pbrt/textures.cpp>
#include <pbrt/materials.cpp>
#include <pbrt/wavefront/surfscatter.cpp>

