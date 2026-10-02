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


set (portableRT_DIR "${PORTABLERT_ROOT}/lib/cmake/portableRT")
find_package (portableRT CONFIG REQUIRED)


if (USE_SYCL)
  message (STATUS "portableRT: ${PORTABLERT_ROOT} (SYCL, DPC++: ${DPCPP_ROOT})")
else ()
  message (STATUS "portableRT: ${PORTABLERT_ROOT} (SYCL unavailable: CPU backends only)")
endif ()

# The imported portableRT target embeds no runtime search path for its own
# libraries, so teach the build (and later the install tree) where they are.
list (APPEND CMAKE_BUILD_RPATH "${PORTABLERT_ROOT}/lib")

# Targets that carry portableRT-generated SYCL kernel libraries link against
# the SYCL runtime with an RPATH of $ORIGIN only (fixed inside portableRT's
# installed macros). Make the SYCL runtime discoverable next to those
# libraries by symlinking it into the target's output directory.
function (prt_add_sycl_runtime_libraries target)
  if (NOT USE_SYCL)
    return ()
  endif ()
  foreach (_prt_lib libportableRT_sycl.so libsycl.so.8 libur_loader.so.0)
    add_custom_command (TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "${PORTABLERT_ROOT}/lib/${_prt_lib}"
              "$<TARGET_FILE_DIR:${target}>/${_prt_lib}")
  endforeach ()
endfunction ()
