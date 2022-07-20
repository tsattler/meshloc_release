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

// Adapts the LO-RANSAC implementation from RansacLib to use a form of effective
// inlier count: rather than counting each 2D-3D match as a separate inlier,
// the query image is tiled into bins and at most one inlier per bin is
// counted.

#ifndef SRC_SPATIAL_RANSAC_H_
#define SRC_SPATIAL_RANSAC_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include <RansacLib/sampling.h>
#include <RansacLib/utils.h>
#include <RansacLib/ransac.h>

namespace ransac_lib {

// Note that the bin size is adjusted in the solver class.
// Note that RANSAC's termination criterion is not necessarily valid anymore.
template <class Model, class ModelVector, class Solver,
          class Sampler = UniformSampling<Solver> >
class LocallyOptimizedEffectiveRANSAC : 
    public LocallyOptimizedMSAC<Model, ModelVector, Solver, Sampler> {
 protected:
  void GetBestEstimatedModelId(const Solver& solver, const ModelVector& models,
                               const int num_models,
                               const double squared_inlier_threshold,
                               double* best_score, int* best_model_id) const {
    *best_score = std::numeric_limits<double>::max();
    *best_model_id = 0;
    for (int m = 0; m < num_models; ++m) {
      double score = 0;
      ScoreModel(solver, models[m], squared_inlier_threshold, &score);

      if (score > *best_score) {
        *best_score = score;
        *best_model_id = m;
      }
    }
  }

  void ScoreModel(const Solver& solver, const Model& model,
                  const double squared_inlier_threshold, double* score) const {
    const int kNumData = solver.num_data();
    *score = 0.0;
    std::unordered_set<int> effective_inliers;
    for (int i = 0; i < kNumData; ++i) {
      int hash;
      double squared_error = solver.EvaluateModelOnPoint(model, i, &hash);
      if (squared_error < squared_inlier_threshold) {
        effective_inliers.emplace(hash);
      }
    }
    *score = static_cast<double>(effective_inliers.size());
  }

  // Standard inlier scroing function.
  inline double ComputeScore(const double squared_error,
                             const double squared_error_threshold) const {
    return (squared_error < squared_error_threshold)? 1.0 : 0.0;
  }

  inline void UpdateBestModel(const double score_curr, const Model& m_curr,
                              double* score_best, Model* m_best) const {
    if (score_curr > *score_best) {
      *score_best = score_curr;
      *m_best = m_curr;
    }
  }
};

}  // namespace ransac_lib

#endif  // SRC_SPATIAL_RANSAC_H_
