// Based on the calibrated absolute pose estimator example provided by 
// RansacLib.

#ifndef MULTI_ABSOLUTE_POSE_ESTIMATOR_H_
#define MULTI_ABSOLUTE_POSE_ESTIMATOR_H_

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

class MultiAbsolutePoseEstimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MultiAbsolutePoseEstimator(const double f_x, const double f_y,
                             const double squared_inlier_threshold,
                             const int width, const int height,
                             const double bin_size,
                             const Points2D& points2D,
                             const ViewingRays& rays,
                             const Points3DVec& points3D,
                             const int num_trials);

  inline int min_sample_size() const { return 4; }

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
  Points3DVec points3D_;
  // Stores the viewing ray for each 2D point position.
  ViewingRays rays_;
  int num_data_;

  int num_trials_;

  mutable std::minstd_rand rng_;
};

}  // namespace visloc_help

#endif  // MULTI_ABSOLUTE_POSE_ESTIMATOR_H_
