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

#include <colmap/base/camera.h>
#include <colmap/estimators/pose.h>
// #include "colmap/util/random.h"

#include <PoseLib/camera_pose.h>
#include <PoseLib/misc/colmap_models.h>
#include <PoseLib/robust/bundle.h>
#include <PoseLib/robust/ransac.h>
#include <PoseLib/robust.h>
#include <PoseLib/types.h>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "absolute_pose_estimator.h"
#include "multi_absolute_pose_estimator.h"
#include "utils.h"
#include "spatial_ransac.h"

namespace py = pybind11;

namespace undistortion {

// Distortion function, implementing various camera models.
void Distortion(const mesh_loc::Camera& camera, const double u,
                const double v, double* du, double* dv) {
  if (camera.camera_type.compare("SIMPLE_RADIAL") == 0) {
    const double kR2 = u * u + v * v;
    const double kRadial = camera.radial[0] * kR2;
    *du = u * kRadial;
    *dv = v * kRadial;
  } else if (camera.camera_type.compare("RADIAL") == 0) {
    const double kR2 = u * u + v * v;
    const double kRadial = camera.radial[0] * kR2 + camera.radial[1] * kR2 * kR2;
    *du = u * kRadial;
    *dv = v * kRadial;
  } else if (camera.camera_type.compare("BROWN_3_PARAMS") == 0) {
    const double kR2 = u * u + v * v;
    const double kRadial = camera.radial[0] * kR2 + camera.radial[1] * kR2 * kR2
                                           + camera.radial[2] * kR2 * kR2 * kR2;
    *du = u * kRadial;
    *dv = v * kRadial;
  } else if (camera.camera_type.compare("PINHOLE") == 0) {
    *du = 0;
    *dv = 0;    
  } else {
    std::cerr << " ERROR: Distortion function for camera model "
              << camera.camera_type << " not yet implemented" << std::endl;
  }
}

// The following code is taken from 
// https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
// Please see the license file for details.
// Assumes that principal point has been subtracted and that the coordinates
// have been divided by the focal length.
void IterativeUndistortion(const mesh_loc::Camera& camera, double* u,
                           double* v) {
  // Parameters for Newton iteration using numerical differentiation with
  // central differences, 100 iterations should be enough even for complex
  // camera models with higher order terms.
  const size_t kNumIterations = 100;
  const double kMaxStepNorm = 1e-10;
  const double kRelStepSize = 1e-6;

  Eigen::Matrix2d J;
  const Eigen::Vector2d x0(*u, *v);
  Eigen::Vector2d x(*u, *v);
  Eigen::Vector2d dx;
  Eigen::Vector2d dx_0b;
  Eigen::Vector2d dx_0f;
  Eigen::Vector2d dx_1b;
  Eigen::Vector2d dx_1f;

  for (size_t i = 0; i < kNumIterations; ++i) {
    const double step0 = std::max(std::numeric_limits<double>::epsilon(),
                                  std::abs(kRelStepSize * x(0)));
    const double step1 = std::max(std::numeric_limits<double>::epsilon(),
                                  std::abs(kRelStepSize * x(1)));
    Distortion(camera, x(0), x(1), &dx(0), &dx(1));
    Distortion(camera, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
    Distortion(camera, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
    Distortion(camera, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
    Distortion(camera, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
    J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
    J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
    J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
    J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
    const Eigen::Vector2d step_x = J.inverse() * (x + dx - x0);
    x -= step_x;
    if (step_x.squaredNorm() < kMaxStepNorm) {
      break;
    }
  }

  *u = x(0);
  *v = x(1);
}

}  // namespace undistortion

py::dict pose_estimation(
  const py::dict query_camera,
  const std::vector<py::dict> db_cams,
  const std::vector<py::dict> matches,
  const py::dict pose_options) {
  // Imports definitions.
  using ransac_lib::LocallyOptimizedMSAC;
  using mesh_loc::AbsolutePoseEstimator;
  using mesh_loc::MultiAbsolutePoseEstimator;
  using mesh_loc::CameraPose;
  using mesh_loc::CameraPoses;
  using mesh_loc::Points2D;
  using mesh_loc::Points3D;
  using mesh_loc::Points3DVec;
  using mesh_loc::ViewingRays;
  using mesh_loc::Camera;
  using mesh_loc::CreateCamera;
  using mesh_loc::triangulate_observations;
  using mesh_loc::select_and_refine_point;
  using mesh_loc::compute_covisibility_clusters;
  using mesh_loc::cluster_keypoints;
  using mesh_loc::get_camera_position;
  using undistortion::IterativeUndistortion;
  using ransac_lib::LocallyOptimizedEffectiveRANSAC;

  std::cout << matches.size() << " " 
            << pose_options["triangulate"].cast<bool>() << " " 
            << pose_options["merge_3D_points"].cast<bool>() << " "
            << pose_options["cluster_keypoints"].cast<bool>() << " " 
            << pose_options["covisibility_filtering"].cast<bool>()
            << " " << pose_options["use_all_matches"].cast<bool>() << " " 
            << " " << pose_options["num_LO_iters"].cast<int>() << " "
            << pose_options["inlier_threshold"].cast<double>() << " "
            << pose_options["min_ransac_iterations"].cast<uint32_t>() << " " 
            << pose_options["max_ransac_iterations"].cast<uint32_t>()
            << std::endl;

  ////
  // Creates the query and database cameras.
  const int kNumDb = static_cast<int>(db_cams.size());
  std::vector<Camera> db_cameras(kNumDb);
  for (int i = 0; i < kNumDb; ++i) {
    CreateCamera(db_cams[i]["model"].cast<std::string>(),
                 db_cams[i]["params"].cast<std::vector<double>>(),
                 db_cams[i]["q"].cast<Eigen::Vector4d>(),
                 db_cams[i]["t"].cast<Eigen::Vector3d>(),
                 &(db_cameras[i]));
  }
  Camera cam;
  CreateCamera(query_camera["model"].cast<std::string>(),
               query_camera["params"].cast<std::vector<double>>(),
               Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
               Eigen::Vector3d::Zero(), &cam);
  const int kWidth = query_camera["width"].cast<int>();
  const int kHeight = query_camera["height"].cast<int>();

  double inlier_threshold = pose_options["inlier_threshold"].cast<double>();
  uint32_t min_num_iterations = 
        pose_options["min_ransac_iterations"].cast<uint32_t>();
  uint32_t max_num_iterations = 
        pose_options["max_ransac_iterations"].cast<uint32_t>();
  int num_LO_iters = pose_options["num_LO_iters"].cast<int>();

  ////
  // Extracts all necessary information from the matches.
  const int kNumInitialMatches = static_cast<int>(matches.size());
  std::vector<Eigen::Vector2d> query_keypoints(kNumInitialMatches);
  std::vector<std::vector<Eigen::Vector3d>> points(kNumInitialMatches);
  std::vector<std::vector<Eigen::Vector2d>> db_observations(kNumInitialMatches);
  std::vector<std::vector<int>> camera_indices(kNumInitialMatches);
  std::cout << " Initial number of matches: " << kNumInitialMatches << std::endl;


  ////
  // Data preparation.
  // std::cout << " Starting to cast" << std::endl;
  for (int i = 0; i < kNumInitialMatches; ++i) {
    // std::cout << " Trying to cast keypoint" << std::endl;
    query_keypoints[i] = matches[i]["keypoint"].cast<Eigen::Vector2d>();
    // std::cout << " Trying to cast points" << std::endl;
    points[i] = matches[i]["points"].cast<std::vector<Eigen::Vector3d>>();
    // std::cout << " Trying to cast observations" << std::endl;
    db_observations[i] = matches[i]["observations"].cast<std::vector<Eigen::Vector2d>>();
    // std::cout << " Trying to cast db_indices" << std::endl;
    camera_indices[i] = matches[i]["db_indices"].cast<std::vector<int>>();

    // if (i == 0) {
    //   std::cout << query_keypoints[i].transpose() << std::endl;
    //   std::cout << points[i].size() << " " << points[i][0].transpose() << std::endl;
    //   std::cout << db_observations[i].size() << " " << db_observations[i][0].transpose() << std::endl;
    //   std::cout << camera_indices[i].size() << " " << camera_indices[i][0] << std::endl;
    // }
  }

  ////
  // Clusters keypoints that are closeby (within 4 pixels). This should only be
  // run in case the features do not have repeatable keypoints, e.g., patch2pix. 
  // if (cluster_keypoints) {
  if (pose_options["cluster_keypoints"].cast<bool>()) {
    std::vector<Eigen::Vector2d> new_keypoints;
    std::vector<std::vector<Eigen::Vector2d>> new_observations;
    std::vector<std::vector<int>> new_camera_indices;
    std::vector<std::vector<Eigen::Vector3d>> new_points;

    cluster_keypoints(db_cameras, query_keypoints, db_observations,
                      camera_indices,  points, 4.0, false, inlier_threshold,
                      kWidth, kHeight, &new_keypoints, &new_observations,
                      &new_camera_indices, &new_points);
    query_keypoints = new_keypoints;
    db_observations = new_observations;
    camera_indices = new_camera_indices;
    points = new_points;
    std::cout << " Found " << query_keypoints.size() << " keypoint clusters"
              << std::endl;
  }

  int num_keypoints = static_cast<int>(query_keypoints.size());

  ////
  // If desired, triangulates 3D points from the database observations.
  // NOTE: At the moment, we assume that there is no distortion in the database
  // images (or that the distortion is negligible).
  if (pose_options["triangulate"].cast<bool>() 
      && pose_options["merge_3D_points"].cast<bool>()) {
    std::cerr << " ERROR: Can either triangulate or merge 3D points, not both."
              << " Decided for merging them" << std::endl;
  }

  // if (merge_3D_points) {
  int count_merged = 0;
  if (pose_options["merge_3D_points"].cast<bool>()) {
    bool using_multiple_points = false;
    for (int i = 0; i < num_keypoints; ++i) {
      Eigen::Vector3d X;
      std::vector<int> inlier_indices;
      int num_consistent = select_and_refine_point(db_cameras,
                            db_observations[i], camera_indices[i],
                            points[i], inlier_threshold, &X, &inlier_indices);
      if (num_consistent < 1) {
        points[i].clear();
        db_observations[i].clear();
        camera_indices[i].clear();
      } else if (num_consistent == 1) {
        using_multiple_points = true;
        ++count_merged;
      } else {
        points[i] = {X};
        std::vector<Eigen::Vector2d> selected_obs;
        std::vector<int> selected_ids;
        for (int id : inlier_indices) {
          selected_obs.push_back(db_observations[i][id]);
          selected_ids.push_back(camera_indices[i][id]);
        }
        db_observations[i] = selected_obs;
        camera_indices[i] = selected_ids;
      }
    }

    if (using_multiple_points) {
      std::cout << " NOTICE: " << count_merged << " query keypoint where "
                << "merging 3D was not possible. Using all 3D points for these "
                << "points." << std::endl;
    }
  } else if (pose_options["triangulate"].cast<bool>()) {
// 
  // } else if (triangulate) {
    for (int i = 0; i < num_keypoints; ++i) {
      Eigen::Vector3d X;
      std::vector<int> inlier_indices;
      int num_consistent = triangulate_observations(db_cameras,
                            db_observations[i], camera_indices[i],
                            inlier_threshold, 1.0, &X, &inlier_indices);
      if (num_consistent <= 1) {
        db_observations[i].clear();
        camera_indices[i].clear();
        points[i].clear();
      } else {
        points[i] = {X};
        std::vector<Eigen::Vector2d> selected_obs;
        std::vector<int> selected_ids;
        for (int id : inlier_indices) {
          selected_obs.push_back(db_observations[i][id]);
          selected_ids.push_back(camera_indices[i][id]);
        }
        db_observations[i] = selected_obs;
        camera_indices[i] = selected_ids;
      }
    }
  }

  ////
  // Performs covisibility filtering.
  std::vector<int> covis_cluster_ids(kNumDb, 0);
  int num_covisility_clusters = 1;
  if (pose_options["covisibility_filtering"].cast<bool>()) {
  // if (covisibility_filtering) {
    num_covisility_clusters = compute_covisibility_clusters(
      db_cameras, db_observations, camera_indices, points, inlier_threshold,
      &covis_cluster_ids);
  }
  std::cout << " Found " << num_covisility_clusters << " covisibility clusters"
            << std::endl;

  ////
  // Pose estimation per covisibility cluster.
  py::dict result_dict;
  result_dict["success"] = false;
  result_dict["num_inliers"] = 0;

  std::vector<Eigen::Vector3d> camera_positions(kNumDb);
  for (int k = 0; k < kNumDb; ++k) {
    camera_positions[k] = get_camera_position(db_cameras[k]);
  }


  for (int c = 0; c < num_covisility_clusters; ++c) {
    Points2D points2D_;
    Points3D points3D_;
    Points2D points2D_aggreg_;
    Points3DVec points3D_aggreg_;
    
    int num_filtered = 0;
    for (int i = 0; i < num_keypoints; ++i) {
      const int kNumObs = static_cast<int>(db_observations[i].size());
      const bool kSinglePoint = points[i].size() == 1u;

      Points3D pts_;

      for (int j = 0; j < kNumObs; ++j) {
        if (covis_cluster_ids[camera_indices[i][j]] == c) {
          // There can be 3D points without a valid depth which survive until
          // this stage, e.g., because we never merge 3D points.
          // Filters these points out as they correspond to 3D points with
          // positions nearly identical to the database image they come from.
          if ((points[i][j] - 
            camera_positions[camera_indices[i][j]]).squaredNorm() <= 0.000001) {
            ++num_filtered;
            continue;
          }
          points2D_.push_back(query_keypoints[i]);
          points3D_.push_back(points[i][j]);
          pts_.push_back(points[i][j]);
          if (kSinglePoint) break;
        }
      }

      if (!pts_.empty()) {
        points2D_aggreg_.push_back(query_keypoints[i]);
        points3D_aggreg_.push_back(pts_);
      }
    }
    std::cout << " Num filtered: " << num_filtered << std::endl;

    const int kNumMatches = static_cast<int>(points2D_.size());
    if (kNumMatches <= 3) continue;

      // Undistorts the keypoints.
    for (int i = 0; i < kNumMatches; ++i) {
      double x = (points2D_[i][0] - cam.c_x) / cam.focal_x;
      double y = (points2D_[i][1] - cam.c_y) / cam.focal_y;
      IterativeUndistortion(cam, &x, &y);
      points2D_[i] << x * cam.focal_x, y * cam.focal_y;
    }

    ViewingRays rays, rays_aggreg;
    AbsolutePoseEstimator::PixelsToViewingRays(
        cam.focal_x, cam.focal_y, points2D_, &rays);

    const int kNumMatchesAggreg = static_cast<int>(points2D_aggreg_.size());
    for (int i = 0; i < kNumMatchesAggreg; ++i) {
      double x = (points2D_aggreg_[i][0] - cam.c_x) / cam.focal_x;
      double y = (points2D_aggreg_[i][1] - cam.c_y) / cam.focal_y;
      IterativeUndistortion(cam, &x, &y);
      points2D_aggreg_[i] << x * cam.focal_x, y * cam.focal_y;
    }
    AbsolutePoseEstimator::PixelsToViewingRays(
        cam.focal_x, cam.focal_y, points2D_aggreg_, &rays_aggreg);

    ransac_lib::LORansacOptions options;
    options.min_num_iterations_ = min_num_iterations;
    options.max_num_iterations_ = max_num_iterations;
    options.min_sample_multiplicator_ = 7;
    options.num_lsq_iterations_ = 4;
    options.num_lo_steps_ = num_LO_iters;
    options.lo_starting_iterations_ = 100u;
    options.final_least_squares_ = true;

    std::random_device rand_dev;
    options.random_seed_ = rand_dev();

    const double kInThreshPX = static_cast<double>(inlier_threshold);
    options.squared_inlier_threshold_ = kInThreshPX * kInThreshPX;

    AbsolutePoseEstimator solver(cam.focal_x, cam.focal_y, 
                                 kInThreshPX * kInThreshPX,
                                 kWidth, kHeight, 16, 
                                 points2D_, rays, points3D_);

    MultiAbsolutePoseEstimator solver_multi(cam.focal_x, cam.focal_y, 
                                            kInThreshPX * kInThreshPX,
                                            kWidth, kHeight, 16, 
                                            points2D_aggreg_, rays_aggreg, 
                                            points3D_aggreg_, 10);


    LocallyOptimizedMSAC<CameraPose, CameraPoses,
                         AbsolutePoseEstimator> lomsac;
    LocallyOptimizedEffectiveRANSAC<CameraPose, CameraPoses,
                                    AbsolutePoseEstimator> loeffsac;
    LocallyOptimizedMSAC<CameraPose, CameraPoses,
                         MultiAbsolutePoseEstimator> multi_lomsac;

    ransac_lib::RansacStatistics ransac_stats;
    CameraPose best_model;

    colmap::Camera colmap_camera;
    colmap_camera.SetModelIdFromName("PINHOLE");
    colmap_camera.SetWidth(kWidth);
    colmap_camera.SetHeight(kHeight);
    std::vector<double> colmap_cam_params = {cam.focal_x, cam.focal_y,
                                             cam.c_x, cam.c_y};
    colmap_camera.SetParams(colmap_cam_params);

    colmap::AbsolutePoseEstimationOptions colmap_abs_pose_options;
    colmap_abs_pose_options.estimate_focal_length = false;
    colmap_abs_pose_options.ransac_options.max_error = kInThreshPX;
    colmap_abs_pose_options.ransac_options.min_inlier_ratio = 0.01;
    colmap_abs_pose_options.ransac_options.min_num_trials = min_num_iterations;
    colmap_abs_pose_options.ransac_options.max_num_trials = max_num_iterations;
    colmap_abs_pose_options.ransac_options.confidence = 0.9999;

    std::vector<char> colmap_inlier_mask;

    colmap::AbsolutePoseRefinementOptions colmap_refinement_options;
    colmap_refinement_options.refine_focal_length = false;
    colmap_refinement_options.refine_extra_params = false;
    colmap_refinement_options.print_summary = false;

    std::string ransac_type = pose_options["ransac_type"].cast<std::string>();
    if (ransac_type.compare("MULTIMSAC") != 0) {
      std::cout << "       running " << ransac_type << " on " << kNumMatches 
                << " matches " << std::endl;
    } else {
      std::cout << "       running " << ransac_type << " on " 
                << kNumMatchesAggreg << " matches " << std::endl;
    }
    auto ransac_start = std::chrono::system_clock::now();

    int num_ransac_inliers = 0;
    if (ransac_type.compare("MSAC") == 0 ||
        ransac_type.compare("MSAC+REF") == 0) {
      num_ransac_inliers = 
          lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);
    } else if (ransac_type.compare("EFFSAC") == 0) {
      num_ransac_inliers = 
          loeffsac.EstimateModel(options, solver, &best_model, &ransac_stats);
    } else if (ransac_type.compare("PYCOLMAP") == 0) {
      

      Eigen::Vector4d q_vec;
      Eigen::Vector3d t_vec;
      size_t num_colmap_inliers;
      for (int i = 0; i < kNumMatches; ++i) {
        points2D_[i] += Eigen::Vector2d(cam.c_x, cam.c_y);
      }

      // std::vector<Eigen::Vector3d> points3D_colmap;
      // points3D_colmap.assign(points3D_.begin(), points3D_.end());
      if (!colmap::EstimateAbsolutePose(colmap_abs_pose_options,
                                        points2D_, points3D_,
                                        &q_vec, &t_vec, &colmap_camera,
                                        &num_colmap_inliers,
                                        &colmap_inlier_mask)) {
        num_ransac_inliers = 0;
      } else {
        // Refines the estimated pose.
        if (!colmap::RefineAbsolutePose(colmap_refinement_options,
                                        colmap_inlier_mask,
                                        points2D_, points3D_,
                                        &q_vec, &t_vec, &colmap_camera)) {
          num_ransac_inliers = 0;
        } else {
          auto ransac_end = std::chrono::system_clock::now();
          std::chrono::duration<double> elapsed_seconds = ransac_end - ransac_start;
          std::cout << "   ... " << ransac_type << " found " 
                    <<  num_colmap_inliers << " inliers" << std::endl;
          std::cout << "   ... " << ransac_type << " took " 
                    << elapsed_seconds.count() << " s" << std::endl;
          result_dict["success"] = true;
          result_dict["qvec"] = q_vec;
          result_dict["tvec"] = t_vec;
          result_dict["num_inliers"] = num_colmap_inliers;
          std::vector<bool> colmap_inliers;
          for (auto it : colmap_inlier_mask) {
            bool inlier = false;
            if (it) inlier = true;
            colmap_inliers.push_back(inlier);
          }
          result_dict["inliers"] = colmap_inliers;
          return result_dict;
        }
      }
    } else if (ransac_type.compare("MULTIMSAC") == 0) {
      num_ransac_inliers = multi_lomsac.EstimateModel(options, solver_multi,
                                                      &best_model, 
                                                      &ransac_stats);
    } else if (ransac_type.compare("POSELIB") == 0 ||
               ransac_type.compare("POSELIB+REF") == 0) {
      poselib::BundleOptions bundle_opts;
      bundle_opts.loss_type = poselib::BundleOptions::LossType::CAUCHY;
      bundle_opts.loss_scale = 1.0;  //inlier_threshold * 0.5;

      poselib::RansacOptions ransac_opts;
      ransac_opts.max_iterations = max_num_iterations;
      ransac_opts.min_iterations = min_num_iterations;
      ransac_opts.max_reproj_error = inlier_threshold;

      // Assumes that keypoints are already centered and undistorted.
      std::vector<double> cam_params = {cam.focal_x, cam.focal_y, 0.0, 0.0};
      poselib::Camera poselib_cam("PINHOLE", cam_params, kWidth, kHeight);
      std::vector<char> inliers_poselib;

      poselib::CameraPose poselib_pose;

      poselib::RansacStats r_stats = poselib::estimate_absolute_pose(
        points2D_, points3D_, poselib_cam, ransac_opts, bundle_opts,
        &poselib_pose, &inliers_poselib);

      num_ransac_inliers = r_stats.num_inliers;
      ransac_stats.num_iterations = r_stats.iterations;
      ransac_stats.inlier_ratio = r_stats.inlier_ratio;

      best_model.topLeftCorner<3, 3>() = poselib_pose.R();
      best_model.col(3) = -best_model.topLeftCorner<3, 3>().transpose() * poselib_pose.t;

      ransac_stats.inlier_indices.clear();
      for (int in = 0; in < static_cast<int>(points2D_.size()); ++in) {
        if (inliers_poselib[in] == true) {
          ransac_stats.inlier_indices.push_back(in);
        }
      }

      ransac_stats.number_lo_iterations = r_stats.refinements;
    }

    if (num_ransac_inliers > 0 && (ransac_type.compare("MSAC") == 0 ||
                                   ransac_type.compare("MSAC+REF") == 0 || 
                                   ransac_type.compare("EFFSAC") == 0)) {
      for (int i = 0; i < kNumMatches; ++i) {
        points2D_[i] += Eigen::Vector2d(cam.c_x, cam.c_y);
      }

      Eigen::Matrix3d R = best_model.topLeftCorner<3, 3>();
      Eigen::Vector3d t = -R * best_model.col(3);
      Eigen::Quaterniond q(R);
      q.normalize();
      Eigen::Vector4d q_vec(q.w(), q.x(), q.y(), q.z());
      Eigen::Vector3d t_vec = t;
      colmap_inlier_mask.resize(kNumMatches, false);
      for (int inl : ransac_stats.inlier_indices) {
        colmap_inlier_mask[inl] = true;
      }

      if (colmap::RefineAbsolutePose(colmap_refinement_options,
                                     colmap_inlier_mask,
                                     points2D_, points3D_,
                                     &q_vec, &t_vec, &colmap_camera)) {
        Eigen::Quaterniond qq;
        qq.w() = q_vec[0];
        qq.x() = q_vec[1];
        qq.y() = q_vec[2];
        qq.z() = q_vec[3];
        best_model.topLeftCorner<3, 3>() = qq.toRotationMatrix();
        best_model.col(3) = -best_model.topLeftCorner<3, 3>().transpose() * t_vec;
      }

      for (int i = 0; i < kNumMatches; ++i) {
        points2D_[i] -= Eigen::Vector2d(cam.c_x, cam.c_y);
      }
    }

    if (num_ransac_inliers < 4) {
      // Default to the pose of the top retrieved image.
      best_model.topLeftCorner<3, 3>() = db_cameras[0].proj_matrix_normalized.topLeftCorner<3, 3>();
      best_model.col(3) = camera_positions[0];
    }

    if (ransac_type.compare("MSAC+REF") == 0 ||
        ransac_type.compare("POSELIB+REF") == 0) {
      int total_num_inliers_xyz = 0;
      Eigen::Vector3d c_weighted(0.0, 0.0, 0.0);
      const double kRange = pose_options["refinement_range"].cast<double>();
      const double kStepSize = pose_options["refinement_step"].cast<double>();
      for (double x = -kRange; x <= kRange; x += kStepSize) {
        for (double y = -kRange; y <= kRange; y += kStepSize) {
          for (double z = -kRange; z <= kRange; z += kStepSize) {
            Eigen::Vector3d c_xyz = best_model.col(3) + Eigen::Vector3d(x, y, z);
            Eigen::Matrix<double, 3, 4> P;
            P.topLeftCorner<3, 3>() = best_model.topLeftCorner<3, 3>();
            P.col(3) = -best_model.topLeftCorner<3, 3>() * c_xyz;
            P.row(0) *= cam.focal_x;
            P.row(1) *= cam.focal_y;
            int num_inliers_xyz = 0;
            for (int i = 0; i < kNumMatches; ++i) {
              Eigen::Vector3d p = P * points3D_[i].homogeneous();
              if (p[2] < 0.0) continue;
              double error = (points2D_[i] - p.hnormalized()).squaredNorm();
              if (error < options.squared_inlier_threshold_) ++num_inliers_xyz;
            }

            total_num_inliers_xyz += num_inliers_xyz;
            c_weighted += c_xyz * static_cast<double>(num_inliers_xyz);
            // std::cout << "   " << num_inliers_xyz << " " << (c_xyz - best_model.col(3)).norm() << " " << num_ransac_inliers << std::endl;
          }
        } 
      }
      c_weighted /= static_cast<double>(total_num_inliers_xyz);
      std::cout << "   distance between BA estimate and weighted position: "
                << (c_weighted - best_model.col(3)).norm() << std::endl;
      if (total_num_inliers_xyz > 0) best_model.col(3) = c_weighted;
    }

    auto ransac_end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = ransac_end - ransac_start;
    std::cout << "   ... " << ransac_type << " found " << num_ransac_inliers 
              << " inliers in " << ransac_stats.num_iterations
              << " iterations with an inlier ratio of "
              << ransac_stats.inlier_ratio << std::endl;
    std::cout << "   ... " << ransac_type << " took " 
              << elapsed_seconds.count() << " s"<< std::endl;
    std::cout << "   ... " << ransac_type << " executed " 
              << ransac_stats.number_lo_iterations
              << " local optimization stages" << std::endl;

    //    if (num_ransac_inliers < 12) continue;

    Eigen::Matrix3d R = best_model.topLeftCorner<3, 3>();
    Eigen::Vector3d t = -R * best_model.col(3);
    Eigen::Quaterniond q(R);
    q.normalize();

    if (num_ransac_inliers > result_dict["num_inliers"].cast<int>()) {
      result_dict["success"] = true;
      result_dict["qvec"] = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
      result_dict["tvec"] = t;
      result_dict["num_inliers"] = num_ransac_inliers;
      result_dict["inliers"] = ransac_stats.inlier_indices;
    }    
  }
  
  return result_dict;
}
