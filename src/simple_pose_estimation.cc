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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <RansacLib/ransac.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "utils.h"
#include "absolute_pose_estimator.h"

namespace py = pybind11;

py::dict simple_pose_estimation(const std::string camera_type,
                                const std::vector<double> camera_intrinsics,
                                const std::vector<Eigen::Vector2d> points2D,
                                const std::vector<Eigen::Vector3d> points3D,
                                const std::vector<Eigen::Vector3d> priors,
                                const double inlier_threshold,
                                const int num_LO_iters,
                                const uint32_t min_num_iterations,
                                const uint32_t max_num_iterations) {
  using ransac_lib::LocallyOptimizedMSAC;
  using mesh_loc::AbsolutePoseEstimator;
  using mesh_loc::CameraPose;
  using mesh_loc::CameraPoses;
  using mesh_loc::Points2D;
  using mesh_loc::Points3D;
  using mesh_loc::ViewingRays;
  using mesh_loc::Camera;
  using mesh_loc::CreateCamera;

  Points2D points2D_(points2D.begin(), points2D.end());
  Points3D points3D_(points3D.begin(), points3D.end());
  Points3D priors_(priors.begin(), priors.end());

  const int kNumMatches = static_cast<int>(points2D.size());

  // Failure output dictionary.
  py::dict result_dict;
  result_dict["success"] = false;
  if (kNumMatches <= 3) {
    return result_dict;
  }

  Camera cam;
  CreateCamera(camera_type, camera_intrinsics, Eigen::Quaterniond::UnitRandom(),
               Eigen::Vector3d::Zero(), &cam);

  // if (camera_type.compare("SIMPLE_RADIAL") == 0) {
  //   cam.focal_x = camera_intrinsics[0];
  //   cam.focal_y = cam.focal_x;
  //   cam.c_x = camera_intrinsics[1];
  //   cam.c_y = camera_intrinsics[2];
  //   cam.radial = {camera_intrinsics[3]};
  //   cam.camera_type = camera_type;
  // } else {
  //   std::cerr << " ERROR: Camera model " << camera_type << " is currently "
  //             << "not supported" << std::endl;
  //   return result_dict;
  // }

  // Undistorts the keypoints.
  for (int i = 0; i < kNumMatches; ++i) {
    double x = (points2D[i][0] - cam.c_x) / cam.focal_x;
    double y = (points2D[i][1] - cam.c_y) / cam.focal_y;
    IterativeUndistortion(cam, &x, &y);
    points2D_[i] << x * cam.focal_x, y * cam.focal_y;
  }

  ViewingRays rays;
  AbsolutePoseEstimator::PixelsToViewingRays(
      cam.focal_x, cam.focal_y, points2D_, &rays);

  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = min_num_iterations;
  options.max_num_iterations_ = max_num_iterations;
  options.min_sample_multiplicator_ = 7;
  options.num_lsq_iterations_ = 4;
  options.num_lo_steps_ = num_LO_iters;
  options.lo_starting_iterations_ = std::min(100u, min_num_iterations / 2u);
  options.final_least_squares_ = true;

  std::random_device rand_dev;
  options.random_seed_ = rand_dev();

  const double kInThreshPX = static_cast<double>(inlier_threshold);
  options.squared_inlier_threshold_ = kInThreshPX * kInThreshPX;

  AbsolutePoseEstimator solver(cam.focal_x, cam.focal_y, 
                               kInThreshPX * kInThreshPX,
                               points2D_, rays, points3D_, priors_);

  LocallyOptimizedMSAC<CameraPose, CameraPoses,
                       AbsolutePoseEstimator>
      lomsac;
  ransac_lib::RansacStatistics ransac_stats;
  CameraPose best_model;

  std::cout << "       running LO-MSAC on " << kNumMatches << " matches " 
            << std::endl;
  auto ransac_start = std::chrono::system_clock::now();

  int num_ransac_inliers =
      lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);
  auto ransac_end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = ransac_end - ransac_start;
  std::cout << "   ... LOMSAC found " << num_ransac_inliers << " inliers in "
            << ransac_stats.num_iterations
            << " iterations with an inlier ratio of "
            << ransac_stats.inlier_ratio << std::endl;
  std::cout << "   ... LOMSAC took " << elapsed_seconds.count() << " s"
            << std::endl;
  std::cout << "   ... LOMSAC executed " << ransac_stats.number_lo_iterations
            << " local optimization stages" << std::endl;

  //    if (num_ransac_inliers < 12) continue;

  Eigen::Matrix3d R = best_model.topLeftCorner<3, 3>();
  Eigen::Vector3d t = -R * best_model.col(3);
  Eigen::Quaterniond q(R);
  q.normalize();

  result_dict["success"] = true;
  result_dict["qvec"] = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
  result_dict["tvec"] = t;
  result_dict["num_inliers"] = num_ransac_inliers;
  result_dict["inliers"] = ransac_stats.inlier_indices;
  
  return result_dict;
}
