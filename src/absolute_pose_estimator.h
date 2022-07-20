// Copyright (c) 2019, Torsten Sattler
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// author: Torsten Sattler, torsten.sattler.de@googlemail.com

// Based on the calibrated absolute pose estimator example provided by 
// RansacLib.

#ifndef ABSOLUTE_POSE_ESTIMATOR_H_
#define ABSOLUTE_POSE_ESTIMATOR_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <PoseLib/solvers/p3p.h>
#include <PoseLib/types.h>

#include "utils.h"
#include "types.h"

namespace mesh_loc {

// using Eigen::Vector3d;
// // An absolute pose is a Eigen 3x4 double matrix storing the rotation and
// // translation of the camera.
// typedef Eigen::Matrix<double, 3, 4> CameraPose;
// typedef std::vector<CameraPose, Eigen::aligned_allocator<CameraPose>>
//     CameraPoses;

// typedef std::vector<Eigen::Vector2d> Points2D;
// typedef std::vector<Vector3d, Eigen::aligned_allocator<Vector3d>> Points3D;
// typedef std::vector<Vector3d, Eigen::aligned_allocator<Vector3d>> ViewingRays;

class AbsolutePoseEstimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AbsolutePoseEstimator(const double f_x, const double f_y,
                        const double squared_inlier_threshold,
                        const int width, const int height,
                        const double bin_size,
                        const Points2D& points2D,
                        const ViewingRays& rays,
                        const Points3D& points3D);

  inline int min_sample_size() const { return 3; }

  inline int non_minimal_sample_size() const { return 6; }

  inline int num_data() const { return num_data_; }

  int MinimalSolver(const std::vector<int>& sample, CameraPoses* poses) const;

  int NonMinimalSolver(const std::vector<int>& sample, CameraPose* pose) const;

  double EvaluateModelOnPoint(const CameraPose& pose, int i) const;

  double EvaluateModelOnPoint(const CameraPose& pose, int i, int* hash) const;

  void LeastSquares(const std::vector<int>& sample, CameraPose* pose) const;

  static void PixelsToViewingRays(const double focal_x, const double focal_y,
                                  const Points2D& points2D, ViewingRays* rays);

 protected:
  // Focal lengths in x- and y-directions.
  double focal_x_;
  double focal_y_;
  double squared_inlier_threshold_;
  // Size of the image.
  int width_;
  int height_;
  double bin_size_;
  int num_x_bins_;
  // Stores the 2D point positions.
  Points2D points2D_;
  // Stores the corresponding 3D point positions.
  Points3D points3D_;
  // Stores the viewing ray for each 2D point position.
  ViewingRays rays_;
  int num_data_;
};

}  // namespace visloc_help

#endif  // ABSOLUTE_POSE_ESTIMATOR_H_
