# @file MLP.R
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
#' @param sizeHidden        Amount of neurons in each default layer, default: 2^(6:9) (64 to 512)
#' @param dropout           How much dropout to apply after first linear, default: seq(0, 0.3, 0.05)
#' @param sizeEmbedding     Size of embedding layer, default: 2^(6:9) (64 to 512)
#' @param estimatorSettings settings of Estimator created with `setEstimator`
#' @param hyperParamSearch  Which kind of hyperparameter search to use random sampling or exhaustive grid search. default: 'random'
#' @param randomSample      How many random samples from hyperparameter space to use
#' @param randomSampleSeed  Random seed to sample hyperparameter combinations
#'
#' @export
setMultiLayerPerceptron <- function(numLayers = as.integer(1:8),
                                    sizeHidden = as.integer(2^(6:9)),
                                    dropout = c(seq(0, 0.3, 0.05)),
                                    sizeEmbedding = as.integer(2^(6:9)),
                                    estimatorSettings = setEstimator(
                                      learningRate = 'auto',
                                      weightDecay = c(1e-6, 1e-3),
                                      batchSize = 1024L,
                                      epochs = 30L,
                                      device="cpu"),
                                    hyperParamSearch = "random",
                                    randomSample = 100,
                                    randomSampleSeed = NULL) {

  paramGrid <- list(
    numLayers = numLayers,
    sizeHidden = sizeHidden,
    dropout = dropout,
    sizeEmbedding = sizeEmbedding
  )

  paramGrid <- c(paramGrid, estimatorSettings$paramsToTune)

  param <- PatientLevelPrediction::listCartesian(paramGrid)
  if (hyperParamSearch == "random" && randomSample>length(param)) {
    stop(paste("\n Chosen amount of randomSamples is higher than the amount of possible hyperparameter combinations.",
               "\n randomSample:", randomSample,"\n Possible hyperparameter combinations:", length(param),
               "\n Please lower the amount of randomSamples"))
  }

  if (hyperParamSearch == "random") {
    suppressWarnings(withr::with_seed(randomSampleSeed, {param <- param[sample(length(param), randomSample)]}))
  }
  attr(param, 'settings')$modelType <- "MLP"

  results <- list(
    fitFunction = "fitEstimator",
    param = param,
    estimatorSettings = estimatorSettings,
    modelType = "MLP",
    saveType = "file",
    modelParamNames = c(
      "numLayers", "sizeHidden",
      "dropout", "sizeEmbedding"
    )
  )

  class(results) <- "modelSettings"

  return(results)
}
