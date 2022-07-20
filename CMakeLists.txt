cmake_minimum_required (VERSION 3.0)
project(meshloc)

#if (EXISTS "${CMAKE_SOURCE_DIR}/cmake")
#  set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
#endif ()

# Same compiler options as PoseLib
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

add_subdirectory (pybind11)
add_subdirectory (src)
