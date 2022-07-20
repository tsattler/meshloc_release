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

#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/StdVector>
#include <PoseLib/camera_pose.h>

#include "absolute_pose_estimator.h"

namespace mesh_loc {

AbsolutePoseEstimator::AbsolutePoseEstimator(
    const double f_x, const double f_y, const double squared_inlier_threshold, 
    const int width, const int height, const double bin_size,
    const Points2D& points2D, const ViewingRays& rays, const Points3D& points3D)
    : focal_x_(f_x),
      focal_y_(f_y),
      squared_inlier_threshold_(squared_inlier_threshold),
      width_(width), height_(height), bin_size_(bin_size), 
      points2D_(points2D),
      points3D_(points3D),
      rays_(rays) {
  num_data_ = static_cast<int>(points2D_.size());
  num_x_bins_ = static_cast<int>(static_cast<double>(width_) / bin_size_ + 0.5);
}

int AbsolutePoseEstimator::MinimalSolver(
    const std::vector<int>& sample, CameraPoses* poses) const {
  poses->clear();
  std::vector<Eigen::Vector3d> x(3), X(3);
  for (int i = 0; i < 3; ++i) {
    x[i] = rays_[sample[i]];
    X[i] = points3D_[sample[i]];
  }

  std::vector<poselib::CameraPose> poselib_poses;
  int num_sols = poselib::p3p(x, X, &poselib_poses);
  // std::cout << num_sols << std::endl;
  if (num_sols == 0) return 0;

  for (const poselib::CameraPose& pose : poselib_poses) {
    CameraPose P;
    P.topLeftCorner<3, 3>() = pose.R();
    P.col(3) = -P.topLeftCorner<3, 3>().transpose() * pose.t;

    poses->push_back(P);

    // const double kError = EvaluateModelOnPoint(P, sample[3]);
    // if (kError < squared_inlier_threshold_) {
      // poses->push_back(P);
      // break;
    // }
  }
  // std::cout << poses->size() << std::endl;

  return static_cast<int>(poses->size());
}

// Returns 0 if no model could be estimated and 1 otherwise.
// Implemented via non-linear optimization in Ceres.
int AbsolutePoseEstimator::NonMinimalSolver(
    const std::vector<int>& sample, CameraPose* pose) const {
  CameraPoses poses;
  std::vector<std::vector<int>> subsamples = 
      {{sample[0], sample[1], sample[2]},
       {sample[1], sample[2], sample[3]},
       {sample[2], sample[3], sample[4]},
       {sample[3], sample[4], sample[5]},
       {sample[0], sample[3], sample[4]},
       {sample[3], sample[3], sample[5]},
       {sample[0], sample[4], sample[5]},
       {sample[1], sample[3], sample[4]},
       {sample[1], sample[3], sample[5]},
       {sample[1], sample[4], sample[5]},
       {sample[2], sample[3], sample[5]},
       {sample[2], sample[4], sample[5]}};
  for (int k = 0; k < 12; ++k) {
    CameraPoses sub_poses;
    if (MinimalSolver(subsamples[k], &sub_poses) > 0) {
      for (const CameraPose& P : sub_poses) {
        poses.emplace_back(P);
      }
    }
  }


  // if (MinimalSolver(sample, &poses) > 0) {

  const int kSampleSize = static_cast<int>(sample.size());
  *pose = poses[0];

  double best_score = std::numeric_limits<double>::max();
  for (const CameraPose& P : poses) {
    double score = 0.0;
    for (int i = 0; i < kSampleSize; ++i) {
      score += std::min(EvaluateModelOnPoint(P, sample[i]),
                        squared_inlier_threshold_);
    }
    if (score < best_score) {
      best_score = score;
      *pose = P;
    }
  }
  
  LeastSquares(sample, pose);
  return 1;
  // } else {
    // return 0;
  // }
}

// Evaluates the pose on the i-th data point.
double AbsolutePoseEstimator::EvaluateModelOnPoint(
    const CameraPose& pose, int i) const {
  Eigen::Vector3d p_c =
      pose.topLeftCorner<3, 3>() * (points3D_[i] - pose.col(3));

  // Check whether point projects behind the camera.
  if (p_c[2] < 0.0) return std::numeric_limits<double>::max();

  Eigen::Vector2d p_2d = p_c.head<2>() / p_c[2];
  p_2d[0] *= focal_x_;
  p_2d[1] *= focal_y_;

  return (p_2d - points2D_[i]).squaredNorm();
}

double AbsolutePoseEstimator::EvaluateModelOnPoint(
    const CameraPose& pose, int i, int* hash) const {
  int hash_x = std::max(0,
    std::min(static_cast<int>(points2D_[i][0] / bin_size_ + 0.5), width_ - 1));
  int hash_y = std::max(0,
    std::min(static_cast<int>(points2D_[i][1] / bin_size_ + 0.5), height_ - 1));
  *hash = hash_x + hash_y * 100000;

  return EvaluateModelOnPoint(pose, i);
}

// Reference implementation using Ceres for refinement.
void AbsolutePoseEstimator::LeastSquares(
    const std::vector<int>& sample, CameraPose* pose) const {
  return;

  Eigen::AngleAxisd aax(pose->topLeftCorner<3, 3>());
  Eigen::Vector3d aax_vec = aax.axis() * aax.angle();
  double* camera = new double[6];
  camera[0] = aax_vec[0];
  camera[1] = aax_vec[1];
  camera[2] = aax_vec[2];
  Eigen::Vector3d t = -pose->topLeftCorner<3, 3>() * pose->col(3);
  camera[3] = t[0];
  camera[4] = t[1];
  camera[5] = t[2];

  ceres::Problem refinement_problem;

  // ceres::LossFunction* loss_fcnt = new ceres::CauchyLoss(1.0);

  const int kSampleSize = static_cast<int>(sample.size());
  for (int i = 0; i < kSampleSize; ++i) {
    const int kIdx = sample[i];
    const Eigen::Vector2d& p_img = points2D_[kIdx];
    const Eigen::Vector3d& p_3D = points3D_[kIdx];
    ceres::CostFunction* cost_function =
        ReprojectionError::CreateCost(p_img[0], p_img[1], p_3D[0], p_3D[1],
                                      p_3D[2], focal_x_, focal_y_);

    refinement_problem.AddResidualBlock(cost_function, nullptr, camera);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &refinement_problem, &summary);

  if (summary.IsSolutionUsable()) {
    Eigen::Vector3d axis(camera[0], camera[1], camera[2]);
    double angle = axis.norm();
    axis.normalize();
    aax.axis() = axis;
    aax.angle() = angle;

    pose->topLeftCorner<3, 3>() = aax.toRotationMatrix();
    t = Eigen::Vector3d(camera[3], camera[4], camera[5]);
    pose->col(3) = -pose->topLeftCorner<3, 3>().transpose() * t;
  }

  delete [] camera;
  camera = nullptr;
}

void AbsolutePoseEstimator::PixelsToViewingRays(
    const double focal_x, const double focal_y, const Points2D& points2D,
    ViewingRays* rays) {
  const int kNumData = static_cast<int>(points2D.size());

  // Creates the bearing vectors and points for the OpenGV adapter.
  rays->resize(kNumData);
  for (int i = 0; i < kNumData; ++i) {
    (*rays)[i] = points2D[i].homogeneous();
    (*rays)[i][0] /= focal_x;
    (*rays)[i][1] /= focal_y;
    (*rays)[i].normalize();
  }
}

}  // namespace visloc_help
