cmake_minimum_required (VERSION 3.0)
project(meshloc)

# Same compiler options as PoseLib
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Enable PyBind11 error messages
add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES)

add_subdirectory (pybind11)
add_subdirectory (src)
