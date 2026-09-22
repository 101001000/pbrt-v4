# Consumer-side integration of the installed portableRT package.
#
# portableRT is consumed exactly as installed (never built from source):
# this module locates the installation, selects which of the prebuilt
# features this project consumes (only SYCL; every other feature is
# explicitly disabled) and provides the runtime paths needed to load the
# portableRT shared objects.

set (PORTABLERT_ROOT
     "${CMAKE_CURRENT_SOURCE_DIR}/../portableRT/build/install"
     CACHE PATH "Path to the installed portableRT package (portableRT/build/install)")
set (DPCPP_ROOT
     ""
     CACHE PATH "Path to the oneAPI DPC++ installation used to build portableRT SYCL kernels")

if (NOT EXISTS "${PORTABLERT_ROOT}/lib/cmake/portableRT/portableRTConfig.cmake")
  message (FATAL_ERROR
           "portableRT installation not found at '${PORTABLERT_ROOT}'.\n"
           "Set -DPORTABLERT_ROOT=<path to portableRT's build/install directory>.")
endif ()

# Select exactly which prebuilt portableRT features this project consumes.
# These must be decided before find_package loads portableRT's Features.cmake.
set (USE_CUDA OFF)
set (USE_OPTIX OFF)
set (USE_HIP OFF)
set (USE_HIPRT OFF)
set (USE_EMBREE OFF)
set (USE_EMBREE_SYCL OFF)
set (USE_SYCL ON)

set (portableRT_DIR "${PORTABLERT_ROOT}/lib/cmake/portableRT")
find_package (portableRT CONFIG REQUIRED)

# The installed kernel macros probe for CUDA/HIP to pick extra SYCL targets;
# pin the target list to the Intel GPU/SPIR-V target that portableRT itself
# was built for.
set (KA_SYCL_COMPILE_TARGETS "spir64" CACHE STRING "portableRT SYCL kernel targets")

if (USE_SYCL)
  message (STATUS "portableRT: ${PORTABLERT_ROOT} (SYCL, DPC++: ${DPCPP_ROOT})")
else ()
  message (STATUS "portableRT: ${PORTABLERT_ROOT} (SYCL unavailable: CPU backends only)")
endif ()

# The imported portableRT target embeds no runtime search path for its own
# libraries, so teach the build (and later the install tree) where they are.
list (APPEND CMAKE_BUILD_RPATH "${PORTABLERT_ROOT}/lib")
