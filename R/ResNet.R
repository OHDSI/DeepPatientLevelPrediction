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
#' Creates settings for the package's default residual-network model.
#'
#' @details
#' The architecture is based on
#' [Gorishniy et al. (2021)](https://arxiv.org/abs/2106.11959). The
#' hyperparameters are defaults selected for patient-level prediction tasks.
#'
#' @param estimatorSettings Estimator settings created by [setEstimator()].
#'
#' @return A `modelSettings` object for use with `PatientLevelPrediction`.
#'
#' @examples
#' resnetSettings <- setDefaultResNet()
#' resnetSettings$param[[1]]$numLayers
#'
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
  resnetSettings <- setResNet(
    numLayers = 6,
    sizeHidden = 512,
    hiddenFactor = 2,
    residualDropout = 0.1,
    hiddenDropout = 0.4,
    sizeEmbedding = 256,
    estimatorSettings = estimatorSettings,
    hyperParamSearch = "random",
    randomSample = 1
  )
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
#'   `hyperParamSearch = "random"`.
#' @param randomSampleSeed Random seed used when sampling combinations.
#'
#' @return A `modelSettings` object for use with `PatientLevelPrediction`.
#'
#' @examples
#' resnetSettings <- setResNet(
#'   numLayers = c(2, 4),
#'   sizeHidden = 128,
#'   hiddenFactor = 2,
#'   residualDropout = 0.1,
#'   hiddenDropout = 0.1,
#'   sizeEmbedding = 64,
#'   randomSample = 2,
#'   randomSampleSeed = 42
#' )
#' @export
setResNet <- function(numLayers = c(1:8),
                      sizeHidden = c(2^(6:10)),
                      hiddenFactor = c(1:4),
                      residualDropout = c(seq(0, 0.5, 0.05)),
                      hiddenDropout = c(seq(0, 0.5, 0.05)),
                      sizeEmbedding = c(2^(6:9)),
                      estimatorSettings =
                        setEstimator(
                          learningRate = "auto",
                          weightDecay = c(1e-6, 1e-3),
                          device = "cpu",
                          batchSize = 1024,
                          epochs = 30,
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

  checkIsClass(hiddenDropout, "numeric")
  checkHigherEqual(hiddenDropout, 0)

  checkIsClass(sizeEmbedding, c("integer", "numeric"))
  checkHigherEqual(sizeEmbedding, 1)

  checkIsClass(hyperParamSearch, "character")

  checkIsClass(randomSample, c("numeric", "integer"))
  checkHigherEqual(randomSample, 1)

  checkIsClass(randomSampleSeed, c("numeric", "integer", "NULL"))

  paramGrid <- list(
    numLayers = numLayers,
    sizeHidden = sizeHidden,
    hiddenFactor = hiddenFactor,
    residualDropout = residualDropout,
    hiddenDropout = hiddenDropout,
    sizeEmbedding = sizeEmbedding
  )

  paramGrid <- c(paramGrid, estimatorSettings$paramsToTune)

  param <- PatientLevelPrediction::listCartesian(paramGrid)

  if (hyperParamSearch == "random" && randomSample > length(param)) {
    stop(paste(
      "\n Chosen amount of randomSamples is higher than the amount of
               possible hyperparameter combinations.", "\n randomSample:",
      randomSample, "\n Possible hyperparameter combinations:",
      length(param), "\n Please lower the amount of randomSamples"
    ))
  }

  if (hyperParamSearch == "random") {
    suppressWarnings(withr::with_seed(randomSampleSeed, {
      param <- param[sample(
        length(param),
        randomSample
      )]
    }))
  }
  results <- list(
    fitFunction = "DeepPatientLevelPrediction::fitEstimator",
    param = param,
    estimatorSettings = estimatorSettings,
    saveType = "file",
    modelParamNames = c("numLayers", "sizeHidden", "hiddenFactor",
                        "residualDropout", "hiddenDropout", "sizeEmbedding"),
    modelType = "ResNet"
  )
  attr(results$param, "settings")$modelType <- results$modelType

  class(results) <- "modelSettings"

  return(results)
}
