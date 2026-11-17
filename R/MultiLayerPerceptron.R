# @file MultiLayerPerceptron.R
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

#' Create Multilayer Perceptron Settings
#'
#' Creates model and hyperparameter-search settings for a multilayer
#' perceptron.
#'
#' @details
#' The defaults define one model. To search hyperparameters, give several
#' values; the example shows the search space that the package used to search
#' by default.
#'
#' @param numLayers Number of hidden layers.
#' @param sizeHidden Number of units in each hidden layer.
#' @param dropout Dropout probability.
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
#' mlpSettings <- setMultiLayerPerceptron()
#'
#' # a random search over 100 of the 1,792 combinations of this space
#' mlpSearch <- setMultiLayerPerceptron(
#'   numLayers = 1:8,
#'   sizeHidden = 2^(6:9),
#'   dropout = seq(0, 0.3, 0.05),
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
#'
#' @export
setMultiLayerPerceptron <- function(numLayers = 3,
                                    sizeHidden = 256,
                                    dropout = 0.1,
                                    sizeEmbedding = 256,
                                    estimatorSettings =
                                      setEstimator(
                                        learningRate = "auto",
                                        weightDecay = 1e-6,
                                        batchSize = 1024,
                                        epochs = 50,
                                        device = "cpu"
                                      ),
                                    hyperParamSearch = "random",
                                    randomSample = 100,
                                    randomSampleSeed = NULL) {
  checkIsClass(numLayers, c("integer", "numeric"))
  checkHigherEqual(numLayers, 1)

  checkIsClass(sizeHidden, c("integer", "numeric"))
  checkHigherEqual(sizeHidden, 1)

  checkIsClass(dropout, c("numeric"))
  checkHigherEqual(dropout, 0)
  checkLower(dropout, 1)

  checkIsClass(sizeEmbedding, c("numeric", "integer"))
  checkHigherEqual(sizeEmbedding, 1)

  paramGrid <- list(
    numLayers = numLayers,
    sizeHidden = sizeHidden,
    dropout = dropout,
    sizeEmbedding = sizeEmbedding
  )

  createDeepModelSettings(
    modelType = "MultiLayerPerceptron",
    paramGrid = paramGrid,
    estimatorSettings = estimatorSettings,
    hyperParamSearch = hyperParamSearch,
    randomSample = randomSample,
    randomSampleSeed = randomSampleSeed
  )
}
