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

#' setMultiLayerPerceptron
#'
#' @description
#' Creates settings for a Multilayer perceptron model
#'
#' @details
#' Model architecture
#'
#'
#' @param numLayers         Number of layers in network, default: 1:8
#' @param sizeHidden        Amount of neurons in each default layer,
#' default: 2^(6:9) (64 to 512)
#' @param dropout           How much dropout to apply after first linear,
#' default: seq(0, 0.3, 0.05)
#' @param sizeEmbedding     Size of embedding default: 2^(6:9) (64 to 512)
#' @param estimatorSettings settings of Estimator created with `setEstimator`
#' @param hyperParamSearch Deprecated. Use PLP `hyperparameterSettings`
#' instead.
#' @param randomSample Deprecated. Use PLP `hyperparameterSettings` instead.
#' @param randomSampleSeed Deprecated. Use PLP `hyperparameterSettings`
#' instead.
#'
#' @export
setMultiLayerPerceptron <- function(numLayers = c(1:8),
                                    sizeHidden = c(2^(6:9)),
                                    dropout = c(seq(0, 0.3, 0.05)),
                                    sizeEmbedding = c(2^(6:9)),
                                    estimatorSettings =
                                      setEstimator(
                                        learningRate = "auto",
                                        weightDecay = c(1e-6, 1e-3),
                                        batchSize = 1024,
                                        epochs = 30,
                                        device = "cpu"
                                      ),
                                    hyperParamSearch = "random",
                                    randomSample = 100,
                                    randomSampleSeed = NULL) {
  legacySearchExplicit <- !missing(hyperParamSearch) ||
    !missing(randomSample) ||
    !missing(randomSampleSeed)
  checkIsClass(numLayers, c("integer", "numeric"))
  checkHigherEqual(numLayers, 1)

  checkIsClass(sizeHidden, c("integer", "numeric"))
  checkHigherEqual(sizeHidden, 1)

  checkIsClass(dropout, c("numeric"))
  checkHigherEqual(dropout, 0)

  checkIsClass(sizeEmbedding, c("numeric", "integer"))
  checkHigherEqual(sizeEmbedding, 1)

  checkIsClass(hyperParamSearch, "character")

  checkIsClass(randomSample, c("numeric", "integer"))
  checkHigherEqual(randomSample, 1)

  checkIsClass(randomSampleSeed, c("numeric", "integer", "NULL"))

  paramGrid <- list(
    numLayers = numLayers,
    sizeHidden = sizeHidden,
    dropout = dropout,
    sizeEmbedding = sizeEmbedding
  )

  paramGrid <- c(paramGrid, estimatorSettings$paramsToTune)

  results <- createDeepModelSettings(
    paramDefinition = paramGrid,
    estimatorSettings = estimatorSettings,
    modelParamNames = c(
      "numLayers", "sizeHidden",
      "dropout", "sizeEmbedding"
    ),
    modelType = "MultiLayerPerceptron",
    hyperParamSearch = hyperParamSearch,
    randomSample = randomSample,
    randomSampleSeed = randomSampleSeed,
    legacySearchExplicit = legacySearchExplicit
  )

  return(results)
}
