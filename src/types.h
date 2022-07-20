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

#ifndef MESH_LOC_TYPES_H_
#define MESH_LOC_TYPES_H_

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

namespace mesh_loc {

// An absolute pose is a Eigen 3x4 double matrix storing the rotation and
// translation of the camera.
typedef Eigen::Matrix<double, 3, 4> CameraPose;
typedef std::vector<CameraPose> CameraPoses;

typedef std::vector<Eigen::Vector2d> Points2D;
typedef std::vector<Eigen::Vector3d> Points3D;
typedef std::vector<Points3D> Points3DVec;
typedef std::vector<Eigen::Vector3d> ViewingRays;

struct Camera {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double c_x;
  double c_y;
  
  double focal_x;
  double focal_y;
  
  std::string camera_type;
  
  std::vector<double> radial;

  Eigen::Quaterniond q;
  Eigen::Vector3d t;

  Eigen::Matrix<double, 3, 4> proj_matrix;
  Eigen::Matrix<double, 3, 4> proj_matrix_normalized;
};

// Code for optimizing a camera pose by optimizing reprojection errors.
struct ReprojectionError {
  ReprojectionError(double x, double y, double X, double Y, double Z,
                    double fx, double fy)
      : point2D_x(x / fx),
        point2D_y(y / fy),
        point3D_X(X),
        point3D_Y(Y),
        point3D_Z(Z),
        f_x(fx),
        f_y(fy) {}

  template <typename T>
  bool operator()(const T* const camera, T* residuals) const {
    // The last three entries are the camera translation.
    T p[3];
    p[0] = static_cast<T>(point3D_X);  // - camera[3];
    p[1] = static_cast<T>(point3D_Y);  // - camera[4];
    p[2] = static_cast<T>(point3D_Z);  // - camera[5];

    // The first three entries correspond to the rotation matrix stored in an
    // angle-axis representation.
    T p_rot[3];
    ceres::AngleAxisRotatePoint(camera, p, p_rot);
    p_rot[0] += camera[3];
    p_rot[1] += camera[4];
    p_rot[2] += camera[5];

    // T x_proj = static_cast<T>(f_x) * p_rot[0] / p_rot[2];
    // T y_proj = static_cast<T>(f_y) * p_rot[1] / p_rot[2];
    T x_proj = p_rot[0] / p_rot[2];
    T y_proj = p_rot[1] / p_rot[2];

    residuals[0] = static_cast<T>(point2D_x) - x_proj;
    residuals[1] = static_cast<T>(point2D_y) - y_proj;

    return true;
  }

  // Factory function
  static ceres::CostFunction* CreateCost(const double x, const double y,
                                         const double X, const double Y,
                                         const double Z, const double fx,
                                         const double fy) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
        new ReprojectionError(x, y, X, Y, Z, fx, fy)));
  }

  // Assumes that the measurement is centered around the principal point.
  // This camera model does not take any radial distortion into account. If
  // radial distortion is present, one should undistort the measurements first.
  double point2D_x;
  double point2D_y;
  // The 3D point position is fixed as we are only interested in refining the
  // camera parameters.
  double point3D_X;
  double point3D_Y;
  double point3D_Z;
  double f_x;
  double f_y;
};

// Code for optimizing a camera pose by optimizing reprojection errors.
// Each 2D keypoint can be represented by one or more 3D point and we compute
// the average residual.
struct MultiReprojectionError {
  MultiReprojectionError(double x, double y,
                         const std::vector<Eigen::Vector3d>& points,
                         double fx, double fy)
      : point2D_x(x),
        point2D_y(y),
        points_(points),
        f_x(fx),
        f_y(fy) {
    num_points_ = static_cast<int>(points.size());
  }

  template <typename T>
  bool operator()(const T* const camera, T* residuals) const {
    residuals[0] = static_cast<T>(0.0);
    residuals[1] = static_cast<T>(0.0);

    T inv_num_points_ = static_cast<T>(1.0 / static_cast<double>(num_points_));

    for (int i = 0; i < num_points_; ++i) {
      // The last three entries are the camera position.
      T p[3];
      p[0] = points_[i][0] - camera[3];
      p[1] = points_[i][1] - camera[4];
      p[2] = points_[i][2] - camera[5];

      // The first three entries correspond to the rotation matrix stored in an
      // angle-axis representation.
      T p_rot[3];
      ceres::AngleAxisRotatePoint(camera, p, p_rot);

      T x_proj = static_cast<T>(f_x) * p_rot[0] / p_rot[2];
      T y_proj = static_cast<T>(f_y) * p_rot[1] / p_rot[2];

      residuals[0] += (static_cast<T>(point2D_x) - x_proj) * inv_num_points_;
      residuals[1] += (static_cast<T>(point2D_y) - y_proj) * inv_num_points_;
    }

    return true;
  }

  // Factory function
  static ceres::CostFunction* CreateCost(
      const double x, const double y, 
      const std::vector<Eigen::Vector3d>& points, const double fx, 
      const double fy) {
    return (new ceres::AutoDiffCostFunction<MultiReprojectionError, 2, 6>(
        new MultiReprojectionError(x, y, points, fx, fy)));
  }

  // Assumes that the measurement is centered around the principal point.
  // This camera model does not take any radial distortion into account. If
  // radial distortion is present, one should undistort the measurements first.
  double point2D_x;
  double point2D_y;
  std::vector<Eigen::Vector3d> points_;
  int num_points_;
  double f_x;
  double f_y;
};

// Non-linear refinement of a 3D point based on reprojection errors.
struct ReprojectionErrorTriangulation {
  ReprojectionErrorTriangulation(const Camera& cam, double u, double v)
      : point2D_x(u),
        point2D_y(v) {
    P.topLeftCorner<3, 3>() = cam.q.toRotationMatrix();
    P.col(3) = cam.t;
    P.row(0) *= cam.focal_x;
    P.row(1) *= cam.focal_y;
  }

  template <typename T>
  bool operator()(const T* const point, T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > X(point);

    const Eigen::Matrix<T, 2, 1> p = (P * X.homogeneous()).hnormalized();

    residuals[0] = static_cast<T>(point2D_x) - p[0];
    residuals[1] = static_cast<T>(point2D_y) - p[1];

    return true;
  }

  // Factory function
  static ceres::CostFunction* CreateCost(const Camera& cam, double u, double v) {
    return (new ceres::AutoDiffCostFunction<ReprojectionErrorTriangulation, 2, 3>(
        new ReprojectionErrorTriangulation(cam, u, v)));
  }

  // Assumes that the measurement is centered around the principal point.
  // This camera model does not take any radial distortion into account. If
  // radial distortion is present, one should undistort the measurements first.
  double point2D_x;
  double point2D_y;
  // The camera pose is fixed as we only want to optimize the 3D point position.
  Eigen::Matrix<double, 3, 4> P;
};

}  // namespace visloc_help


#endif  // MESH_LOC_TYPES_H_

