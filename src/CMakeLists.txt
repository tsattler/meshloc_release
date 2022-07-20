cmake_minimum_required (VERSION 3.6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(PYBIND11_CPP_STANDARD -std=c++17)

find_package (Eigen3 REQUIRED)

find_package (Ceres REQUIRED)

find_package (COLMAP REQUIRED)

include_directories (
  ${CMAKE_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/RansacLib
  ${CMAKE_SOURCE_DIR}/PoseLib
  ${CMAKE_SOURCE_DIR}/src
  ${COLMAP_INCLUDE_DIRS}
)

link_directories(${COLMAP_LINK_DIRS})

file(GLOB POSELIB_SRC
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/misc/*.h"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/misc/*.cc"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/*.h"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/*.cc"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/solvers/*.h"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/solvers/*.cc"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/*.h"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/*.cc"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/estimators/*.h"
     "${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/estimators/*.cc"     
)

pybind11_add_module(meshloc main.cc absolute_pose_estimator.h absolute_pose_estimator.cc multi_absolute_pose_estimator.h multi_absolute_pose_estimator.cc utils.h utils.cc ${POSELIB_SRC})

# pybind11_add_module(meshloc main.cc absolute_pose_estimator.h absolute_pose_estimator.cc multi_absolute_pose_estimator.h multi_absolute_pose_estimator.cc utils.h utils.cc ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/types.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/camera_pose.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/solvers/p3p.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/solvers/p3p.cc ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/misc/colmap_models.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/misc/colmap_models.cc ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/bundle.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/bundle.cc ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/ransac.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/ransac.cc ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/utils.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/utils.cc ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/sampling.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust/sampling.cc ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust.h ${CMAKE_SOURCE_DIR}/PoseLib/PoseLib/robust.cc)
#pybind11_add_module(meshloc main.cc)
target_link_libraries(meshloc PRIVATE ${CERES_LIBRARIES} ${COLMAP_LIBRARIES})
