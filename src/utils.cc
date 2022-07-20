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
#include "utils.h"

namespace mesh_loc {

// Creates a Camera object from the specifications of a Colmap camera.
// Note that not all camera models are currently supported.
void CreateCamera(const std::string& camera_model,
                  const std::vector<double>& params,
                  const Eigen::Vector4d& q,
                  const Eigen::Vector3d& t,
                  Camera* cam) {
  if (camera_model.compare("SIMPLE_RADIAL") == 0 ||
      camera_model.compare("SIMPLE_RADIAL_FISHEYE") == 0) {
    cam->focal_x = params[0];
    cam->focal_y = params[0];
    cam->c_x = params[1];
    cam->c_y = params[2];
    cam->radial = {params[3]};
    cam->camera_type = camera_model;
  } else if (camera_model.compare("SIMPLE_PINHOLE") == 0) {
    cam->focal_x = params[0];
    cam->focal_y = params[0];
    cam->c_x = params[1];
    cam->c_y = params[2];
    cam->radial.clear();
    cam->camera_type = camera_model;
  } else if (camera_model.compare("PINHOLE") == 0) {
    cam->focal_x = params[0];
    cam->focal_y = params[1];
    cam->c_x = params[2];
    cam->c_y = params[3];
    cam->radial.clear();
    cam->camera_type = camera_model;
  } else if (camera_model.compare("RADIAL") == 0 || 
             camera_model.compare("RADIAL_FISHEYE") == 0) {
    cam->focal_x = params[0];
    cam->focal_y = params[0];
    cam->c_x = params[1];
    cam->c_y = params[2];
    cam->radial = {params[3], params[4]};
    cam->camera_type = camera_model;
  } else {
    std::cerr << " ERROR: Camera model " << camera_model << " is currently "
              << "not supported" << std::endl;
    return;
  }
  cam->q.w() = q[0];
  cam->q.x() = q[1];
  cam->q.y() = q[2];
  cam->q.z() = q[3];
  cam->t = t;
  cam->proj_matrix.topLeftCorner<3, 3>() = cam->q.toRotationMatrix();
  // std::cout << cam->proj_matrix.topLeftCorner<3, 3>() << std::endl << std::endl;
  cam->proj_matrix.col(3) = t;
  cam->proj_matrix_normalized = cam->proj_matrix;
  cam->proj_matrix.row(0) *= cam->focal_x;
  cam->proj_matrix.row(1) *= cam->focal_y;
}

Eigen::Vector2d project(const Camera& cam, const Eigen::Vector3d& X) {
  Eigen::Vector3d p = cam.proj_matrix * X.homogeneous();
  if (p[2] <= 0.0) return Eigen::Vector2d(1000000.0, 1000000.0);
  return p.hnormalized();
  // return (cam.proj_matrix * X.homogeneous()).hnormalized();
}

Eigen::Vector2d project(const Camera& cam, const Eigen::Vector4d& X) {
  Eigen::Vector3d p = cam.proj_matrix * X;
  if (p[2] <= 0.0) return Eigen::Vector2d(1000000.0, 1000000.0);
  return p.hnormalized();
  // return (cam.proj_matrix * X).hnormalized();
}

Eigen::Vector3d get_camera_position(const Camera& cam) {
  return -cam.proj_matrix_normalized.topLeftCorner<3, 3>().transpose()
                                          * cam.proj_matrix_normalized.col(3);
}



// Code for triangulating a 3D point from two observations. See Hartley &
// Zisserman, 2nd edition, Chapter 12.2 (page 312).
void triangulate(const Eigen::Matrix<double, 3, 4>& P1,
                 const Eigen::Matrix<double, 3, 4>& P2,
                 const Eigen::Vector3d& x1, const Eigen::Vector3d& x2,
                 Eigen::Vector3d* X) {
  Eigen::Matrix4d A;
  A.row(0) = x1[0] * P1.row(2) - x1[2] * P1.row(0);
  A.row(1) = x1[1] * P1.row(2) - x1[2] * P1.row(1);
  A.row(2) = x2[0] * P2.row(2) - x2[2] * P2.row(0);
  A.row(3) = x2[1] * P2.row(2) - x2[2] * P2.row(1);

  Eigen::JacobiSVD<Eigen::Matrix<double, 4, 4>> svd(A, Eigen::ComputeFullV);
  Eigen::Vector4d v = svd.matrixV().col(3);
  (*X) = v.hnormalized();
}

int refine_point(const std::vector<Camera>& cameras,
                 const std::vector<Eigen::Vector2d>& observations,
                 const std::vector<int>& camera_indices,
                 const double reprojection_error,
                 Eigen::Vector3d* X, std::vector<int>* inlier_indices) {
  const int kNumObs = static_cast<int>(observations.size());
  const double kSquaredThreshold = reprojection_error * reprojection_error;

  inlier_indices->clear();
  ceres::Problem refinement_problem;
  double* point = new double[3];
  for (int i = 0; i < 3; ++i) point[i] = (*X)[i];

  int num_inliers = 0;
  for (int k = 0; k < kNumObs; ++k) {
    const int kCamId = camera_indices[k];
    Eigen::Vector2d p =  project(cameras[kCamId], *X);
    double error = (p - observations[k]).squaredNorm();

    if (error < kSquaredThreshold) {
      ++num_inliers;
      inlier_indices->push_back(k);
      ceres::CostFunction* cost_function =
          ReprojectionErrorTriangulation::CreateCost(cameras[kCamId],
              observations[k][0], observations[k][1]);
      refinement_problem.AddResidualBlock(cost_function, nullptr, point);
    }
  }

  if (inlier_indices->size() < 2u) {
    delete [] point;
    point = nullptr;
    return static_cast<int>(inlier_indices->size());
  }
  // std::cout << std::endl << X->transpose() << std::endl;

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &refinement_problem, &summary);

  if (summary.IsSolutionUsable()) {
    Eigen::Vector3d Y;
    Y << point[0], point[1], point[2];

    int new_num_inliers = 0;
    std::vector<int> new_inliers;
    for (int k = 0; k < kNumObs; ++k) {
    const int kCamId = camera_indices[k];
      Eigen::Vector2d p =  project(cameras[kCamId], Y);
      double error = (p - observations[k]).squaredNorm();

      if (error < kSquaredThreshold) {
        ++new_num_inliers;
        new_inliers.push_back(k);
      }
    }

    if (new_num_inliers >= num_inliers) {
      *X = Y;
      *inlier_indices = new_inliers;
    }
  }
  delete [] point;
  point = nullptr;
  // std::cout << X->transpose() << std::endl;

  return static_cast<int>(inlier_indices->size());
}

// Given a set of observations in images, computes the corresponding 3D point.  
// Returns the number of inliers for the generated point. If the return value is
// smaller than 2, then the resulting 3D point is not well-defined.
int triangulate_observations(const std::vector<Camera>& cameras,
                             const std::vector<Eigen::Vector2d>& observations,
                             const std::vector<int>& camera_indices,
                             const double reprojection_error,
                             const double min_angle,
                             Eigen::Vector3d* X,
                             std::vector<int>* inlier_indices) {
  if (observations.size() < 2u) return 0;

  const int kNumObs = static_cast<int>(observations.size());
  std::vector<Eigen::Matrix<double, 3, 4>> projection_matrices(kNumObs);
  std::vector<Eigen::Vector3d> rays(kNumObs);
  std::vector<Eigen::Vector3d> global_rays(kNumObs);
  for (int i = 0; i < kNumObs; ++i) {
    projection_matrices[i] = cameras[camera_indices[i]].proj_matrix_normalized;

    rays[i] << observations[i][0] / cameras[camera_indices[i]].focal_x,
               observations[i][1] / cameras[camera_indices[i]].focal_y, 1.0;
    global_rays[i] = 
          projection_matrices[i].topLeftCorner<3, 3>().transpose() * rays[i];
    global_rays[i].normalize();
  }

  // Exhaustively tries all possible combinations.
  const double kAngleThresh = std::cos(min_angle * M_PI / 180.0);
  const double kSquaredThreshold = reprojection_error * reprojection_error;
  double best_score = std::numeric_limits<double>::max();
  for (int i = 0; i < kNumObs; ++i) {
    for (int j = i + 1; j < kNumObs; ++j) {
      double cos_angle = std::fabs(global_rays[i].dot(global_rays[j]));
      if (cos_angle >= kAngleThresh) continue;
      Eigen::Vector3d x;
      triangulate(projection_matrices[i], projection_matrices[j],
                  rays[i], rays[j], &x);

      double score = 0.0;
      Eigen::Vector4d x_ = x.homogeneous();
      for (int k = 0; k < kNumObs; ++k) {
        Eigen::Vector2d p = project(cameras[camera_indices[k]], x_);

        score += std::min(kSquaredThreshold,
                          (p - observations[k]).squaredNorm());
      }

      if (score < best_score) {
        best_score = score;
        *X = x;
      }
    }
  }

  // Non-linear refinement.
  return refine_point(cameras, observations, camera_indices, reprojection_error,
                      X, inlier_indices);
}

// Given a set of 2D observations selects a single 3D point that is consistent 
// with most other observations. Refines the selected point by optimizing the
// reprojection error. Returns the number of observations it is consistent with.
int select_and_refine_point(const std::vector<Camera>& cameras,
                            const std::vector<Eigen::Vector2d>& observations,
                            const std::vector<int>& camera_indices,
                            const std::vector<Eigen::Vector3d>& points,
                            const double reprojection_error,
                            Eigen::Vector3d* X, 
                            std::vector<int>* inlier_indices) {
  const int kNumObs = static_cast<int>(observations.size());

  double best_score = std::numeric_limits<double>::max();
  const double kSquaredThreshold = reprojection_error * reprojection_error;

  for (int i = 0; i < kNumObs; ++i) {
    double score = 0.0;
    for (int k = 0; k < kNumObs; ++k) {
      Eigen::Vector2d p = project(cameras[camera_indices[k]], points[i]);

      score += std::min(kSquaredThreshold, (p - observations[k]).squaredNorm());
    }

    if (score < best_score) {
      *X = points[i];
      best_score = score;
    }
  }

  // Non-linear refinement.
  return refine_point(cameras, observations, camera_indices, reprojection_error,
                      X, inlier_indices);
}


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
  std::vector<int>* cluster_ids_per_cam) {

  int num_cams = static_cast<int>(cameras.size());

  Eigen::MatrixXi A(num_cams, num_cams);
  A.setZero();

  const int kNumMatches = static_cast<int>(observations.size());
  const double kSquaredReprojError = reprojection_error * reprojection_error;
  for (int i = 0; i < kNumMatches; ++i) {
    const int kNumObs = static_cast<int>(observations[i].size());

    for (int j = 0; j < kNumObs; ++j) {
      for (int k = j + 1; k < kNumObs; ++k) {
        if (A(camera_indices[i][j], camera_indices[i][k]) == 1) continue;

        Eigen::Vector2d p1 = project(cameras[camera_indices[i][j]],
                                     points[i][k]);
        Eigen::Vector2d p2 = project(cameras[camera_indices[i][k]],
                                     points[i][j]);
        double max_error = std::max(
          (p1 - observations[i][j]).squaredNorm(), 
          (p2 - observations[i][k]).squaredNorm());
        if (max_error > kSquaredReprojError) continue;
        A(camera_indices[i][j], camera_indices[i][k]) = 1;
        A(camera_indices[i][k], camera_indices[i][j]) = 1;
      }
    }
  }

  std::vector<int>& c_ids = *cluster_ids_per_cam;
  c_ids.resize(num_cams);
  std::fill(c_ids.begin(), c_ids.end(), -1);
  int current_cid = 0;

  for (int i = 0; i < num_cams; ++i) {
    if (c_ids[i] > -1) continue;

    std::queue<int> queue;
    queue.push(i);

    while (!queue.empty()) {
      int current_cam = queue.front();
      queue.pop();

      c_ids[current_cam] = current_cid;

      for (int j = 0; j < num_cams; ++j) {
        if (A(current_cid, j) == 1 && c_ids[j] == -1) {
          queue.push(j);
        }
      }
    }

    ++current_cid;
  }

  return current_cid;
}

// Clusters keypoint detections in the image. This is useful for methods such as
// patch2pix, which do not compute repeatable keypoint positions.
// This approach is inspired by the clustering method from
// [Zhou et al., Patch2Pix: Epipolar-Guided Pixel-Level Correspondences, 
//  CVPR 2021].
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
  std::vector<std::vector<Eigen::Vector3d>>* new_points) {
  new_keypoints->clear();
  new_observations->clear();
  new_camera_indices->clear();
  new_points->clear();

  const int kNumObs = static_cast<int>(observations.size());

  // To efficiently find potential candidates for merging for each keypoint,
  // we use a regular grid (stored as a hash map) with a cell size corresponding
  // to the radius at which points can be merged.
  const double kGridSize = static_cast<double>(
          static_cast<int>(distance_thresh + 0.5));
  // const int kNumXGrids = static_cast<int>(static_cast<float>(width)
                                         // / static_cast<float>(kGridSize) + 0.5);
  std::unordered_multimap<int, int> grid;

  const int kGridMult = 100000;

  // Fills the map.
  for (int i = 0; i < kNumObs; ++i) {
    int cell_x = static_cast<int>(keypoints[i][0] / kGridSize);
    int cell_y = static_cast<int>(keypoints[i][1] / kGridSize);
    grid.emplace(std::make_pair(cell_y * kGridMult + cell_x, i));
  }

  // For each keypoint, uses the map to find relevant points and to compute a
  // score (which is how many points it covers).
  typedef std::pair<int, std::pair<int, double>> ScoreEntry;
  std::vector<ScoreEntry> scores(kNumObs);
  std::vector<std::vector<int>> covers(kNumObs);
  const double kSquaredReprojError = reprojection_error * reprojection_error;
  for (int i = 0; i < kNumObs; ++i) {
    int cell_x = static_cast<int>(keypoints[i][0] / kGridSize);
    int cell_y = static_cast<int>(keypoints[i][1] / kGridSize);
    int score = 0;
    double sum_dist = 0.0;
    covers[i].clear();
    scores[i].first = i;
    for (int x = -1; x <= 1; ++x) {
      for (int y = -1; y <= 1; ++y) {
        int c_x = cell_x + x;
        int c_y = cell_y + y;
        auto indices = grid.equal_range(c_y * kGridMult + c_x);
        for (auto idx = indices.first; idx != indices.second; ++idx) {
          const int kKeyId = idx->second;
          double dist = (keypoints[i] - keypoints[kKeyId]).norm();
          if (dist < distance_thresh) {
            if (use_3D_points) {
              // This should be deprecated.
              Eigen::Vector2d p1 = project(cameras[camera_indices[kKeyId][0]],
                                           points[i][0]);
              Eigen::Vector2d p2 = project(cameras[camera_indices[i][0]],
                                           points[kKeyId][0]);
              double max_error = std::max(
                (p1 - observations[kKeyId][0]).squaredNorm(), 
                (p2 - observations[i][0]).squaredNorm());
              if (max_error > kSquaredReprojError) continue;
            }
            ++score;
            sum_dist += dist;
            covers[i].emplace_back(idx->second);
          }
        }
      }
    }
    scores[i].second.first = score;
    scores[i].second.second = sum_dist;
  }

  // Greedily select keypoints that cover as many 
  std::sort(scores.begin(), scores.end(),
            [](const ScoreEntry& a, const ScoreEntry& b) {
              if (a.second.first == b.second.first) {
                return a.second.second < b.second.second;
              }
              return a.second.first > b.second.first;
            });

  std::vector<bool> covered(kNumObs, false);
  // std::vector<Eigen::Vector2d> keys;
  for (int i = 0; i < kNumObs; ++i) {
    const int kKeyId = scores[i].first;
    if (covered[kKeyId]) continue;

    Eigen::Vector2d key_pt(0.0, 0.0);

    std::vector<Eigen::Vector2d> obs;
    std::vector<int> cam_ids;
    std::vector<Eigen::Vector3d> pts;
    for (const int id : covers[kKeyId]) {
      key_pt += keypoints[id];
      // keys.push_back(keypoints[id]);
      // obs.push_back(observations[id][0]);
      obs.insert(obs.end(), observations[id].begin(),
                 observations[id].end());
      // cam_ids.push_back(camera_indices[id][0]);
      cam_ids.insert(cam_ids.end(), camera_indices[id].begin(),
                     camera_indices[id].end());
      // pts.push_back(points[id][0]);
      pts.insert(pts.end(), points[id].begin(), points[id].end());
      covered[id] = true;
    }
    // Use the average keypoint.
    key_pt /= static_cast<double>(obs.size());

    // new_keypoints->push_back(keypoints[kKeyId]);
    new_keypoints->push_back(key_pt);
    new_observations->push_back(obs);
    new_points->push_back(pts);
    new_camera_indices->push_back(cam_ids);
  }
}

}  // namespace visloc_help


#endif  // MESH_LOC_UTILS_H_

