# @file ResNet.R
#
# Copyright 2022 Observational Health Data Sciences and Informatics
#
# This file is part of DeepPatientLevelPrediction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#' Create Default ResNet Settings
#'
#' Deprecated: [setResNet()] now defaults to the same model.
#'
#' @param estimatorSettings Estimator settings created by [setEstimator()].
#'
#' @return A `modelSettings` object for use with `PatientLevelPrediction`.
#'
#' @keywords internal
#' @export
setDefaultResNet <- function(estimatorSettings =
                               setEstimator(
                                 learningRate = "auto",
                                 weightDecay = 1e-6,
                                 device = "cpu",
                                 batchSize = 1024,
                                 epochs = 50,
                                 seed = NULL
                               )) {
  warning(
    "`setDefaultResNet()` is deprecated. Use `setResNet()`, which now ",
    "defaults to the same model.",
    call. = FALSE
  )
  resnetSettings <- setResNet(estimatorSettings = estimatorSettings)
  attr(resnetSettings, "settings")$name <- "defaultResnet"
  return(resnetSettings)
}


#' Create ResNet Settings
#'
#' Creates model and hyperparameter-search settings for a residual network.
#'
#' @details
#' The architecture is based on
#' [Gorishniy et al. (2021)](https://arxiv.org/abs/2106.11959).
#'
#' The defaults define one model, selected for patient-level prediction tasks.
#' To search hyperparameters, give several values; the example shows the
#' search space that the package used to search by default.
#'
#' @param numLayers Number of residual layers.
#' @param sizeHidden Width of the hidden representation.
#' @param hiddenFactor Multiplier controlling the inner width of each residual
#'   layer.
#' @param residualDropout Dropout probability after the final linear operation
#'   in each residual layer.
#' @param hiddenDropout Dropout probability after the first linear operation in
#'   each residual layer.
#' @param sizeEmbedding Embedding dimension.
#' @param estimatorSettings Estimator settings created by [setEstimator()].
#' @param hyperParamSearch Hyperparameter-search strategy, either `"random"`
#'   or `"grid"`.
#' @param randomSample Number of combinations sampled when
#'   `hyperParamSearch = "random"`. If there are fewer combinations, each is
#'   used once.
#' @param randomSampleSeed Random seed used when sampling combinations.
#'
#' @return A `modelSettings` object for use with `PatientLevelPrediction`.
#'
#' @examples
#' # one model, with the default hyperparameters
#' resnetSettings <- setResNet()
#'
#' # a random search over 100 of the 154,880 combinations of this space
#' resnetSearch <- setResNet(
#'   numLayers = 1:8,
#'   sizeHidden = 2^(6:10),
#'   hiddenFactor = 1:4,
#'   residualDropout = seq(0, 0.5, 0.05),
#'   hiddenDropout = seq(0, 0.5, 0.05),
#'   sizeEmbedding = 2^(6:9),
#'   estimatorSettings = setEstimator(
#'     learningRate = "auto",
#'     weightDecay = c(1e-6, 1e-3),
#'     batchSize = 1024,
#'     epochs = 30
#'   ),
#'   hyperParamSearch = "random",
#'   randomSample = 100,
#'   randomSampleSeed = 42
#' )
#' @export
setResNet <- function(numLayers = 6,
                      sizeHidden = 512,
                      hiddenFactor = 2,
                      residualDropout = 0.1,
                      hiddenDropout = 0.4,
                      sizeEmbedding = 256,
                      estimatorSettings =
                        setEstimator(
                          learningRate = "auto",
                          weightDecay = 1e-6,
                          device = "cpu",
                          batchSize = 1024,
                          epochs = 50,
                          seed = NULL
                        ),
                      hyperParamSearch = "random",
                      randomSample = 100,
                      randomSampleSeed = NULL) {
  checkIsClass(numLayers, c("integer", "numeric"))
  checkHigherEqual(numLayers, 1)

  checkIsClass(sizeHidden, c("integer", "numeric"))
  checkHigherEqual(sizeHidden, 1)

  checkIsClass(residualDropout, "numeric")
  checkHigherEqual(residualDropout, 0)
  checkLower(residualDropout, 1)

  checkIsClass(hiddenDropout, "numeric")
  checkHigherEqual(hiddenDropout, 0)
  checkLower(hiddenDropout, 1)

  checkIsClass(sizeEmbedding, c("integer", "numeric"))
  checkHigherEqual(sizeEmbedding, 1)

  paramGrid <- list(
    numLayers = numLayers,
    sizeHidden = sizeHidden,
    hiddenFactor = hiddenFactor,
    residualDropout = residualDropout,
    hiddenDropout = hiddenDropout,
    sizeEmbedding = sizeEmbedding
  )

  createDeepModelSettings(
    modelType = "ResNet",
    paramGrid = paramGrid,
    estimatorSettings = estimatorSettings,
    hyperParamSearch = hyperParamSearch,
    randomSample = randomSample,
    randomSampleSeed = randomSampleSeed
  )
}
