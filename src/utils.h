// Copyright (c) 2022, Vojtech Panek and Zuzana Kukelova and Torsten Sattler
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
// THE POSSIBILITY OF SUCH DAMAGE.

#ifndef MESH_LOC_UTILS_H_
#define MESH_LOC_UTILS_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "types.h"

namespace mesh_loc {

// Creates a Camera object from the specifications of a Colmap camera.
// Note that not all camera models are currently supported.
void CreateCamera(const std::string& camera_model,
                  const std::vector<double>& params,
                  const Eigen::Vector4d& q,
                  const Eigen::Vector3d& t,
                  Camera* cam);

Eigen::Vector2d project(const Camera& cam, const Eigen::Vector3d& X);

Eigen::Vector2d project(const Camera& cam, const Eigen::Vector4d& X);

Eigen::Vector3d get_camera_position(const Camera& cam);

// Code for triangulating a 3D point from two observations. See Hartley &
// Zisserman, 2nd edition, Chapter 12.2 (page 312).
void triangulate(const Eigen::Matrix<double, 3, 4>& P1,
                 const Eigen::Matrix<double, 3, 4>& P2,
                 const Eigen::Vector3d& x1, const Eigen::Vector3d& x2,
                 Eigen::Vector3d* X);

int refine_point(const std::vector<Camera>& cameras,
                 const std::vector<Eigen::Vector2d>& observations,
                 const std::vector<int>& camera_indices,
                 const double reprojection_error,
                 Eigen::Vector3d* X, std::vector<int>* inlier_indices);

// Given a set of observations in images, computes the corresponding 3D point.  
// Returns the number of inliers for the generated point. If the return value is
// smaller than 2, then the resulting 3D point is not well-defined.
int triangulate_observations(const std::vector<Camera>& cameras,
                             const std::vector<Eigen::Vector2d>& observations,
                             const std::vector<int>& camera_indices,
                             const double reprojection_error,
                             const double min_angle,
                             Eigen::Vector3d* X,
                             std::vector<int>* inlier_indices);

// Given a set of 2D observations selects a single 3D point that is consistent 
// with most other observations. Refines the selected point by optimizing the
// reprojection error. Returns the number of observations it is consistent with.
int select_and_refine_point(const std::vector<Camera>& cameras,
                            const std::vector<Eigen::Vector2d>& observations,
                            const std::vector<int>& camera_indices,
                            const std::vector<Eigen::Vector3d>& points,
                            const double reprojection_error,
                            Eigen::Vector3d* X, 
                            std::vector<int>* inlier_indices);


// Given a set of 2D-3D matches (represented by the indices of the database
// images in which the 3D points were observed), computes the corresponding
// co-visibility graph and detects connected components in the graph.
// Returns the number of clusters.
int compute_covisibility_clusters(
  const std::vector<Camera>& cameras,
  const std::vector<std::vector<Eigen::Vector2d>>& observations,
  const std::vector<std::vector<int>>& camera_indices,
  const std::vector<std::vector<Eigen::Vector3d>>& points,
  const double reprojection_error,
  std::vector<int>* cluster_ids_per_cam);

// Clusters keypoint detections in the image. This is useful for methods such as
// patch2pix, which do not compute repeatable keypoint positions.
// This approach is inspired by the clustering method from
// [Zhou et al., Patch2Pix: Epipolar-Guided Pixel-Level Correspondences, 
//  CVPR 2021].
// Assumes that there is exactly one database keypoint observation and one
// corresponding 3D point per query image keypoint.
void cluster_keypoints(
  const std::vector<Camera>& cameras,
  const std::vector<Eigen::Vector2d>& keypoints,
  const std::vector<std::vector<Eigen::Vector2d>>& observations,
  const std::vector<std::vector<int>>& camera_indices,
  const std::vector<std::vector<Eigen::Vector3d>>& points,
  const double distance_thresh, const bool use_3D_points,
  const double reprojection_error, const int width, const int height,
  std::vector<Eigen::Vector2d>* new_keypoints,
  std::vector<std::vector<Eigen::Vector2d>>* new_observations,
  std::vector<std::vector<int>>* new_camera_indices,
  std::vector<std::vector<Eigen::Vector3d>>* new_points);

}  // namespace visloc_help


#endif  // MESH_LOC_UTILS_H_

