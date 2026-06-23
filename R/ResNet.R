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

#' setDefaultResNet
#'
#' @description
#' Creates settings for a default ResNet model
#'
#' @details
#' Model architecture from by https://arxiv.org/abs/2106.11959 .
#' Hyperparameters chosen by a experience on a few prediction problems.
#'
#' @param estimatorSettings created with ```setEstimator```

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
  oldOption <- getOption("DeepPatientLevelPrediction.suppressLegacySearchWarning")
  options(DeepPatientLevelPrediction.suppressLegacySearchWarning = TRUE)
  on.exit(options(
    DeepPatientLevelPrediction.suppressLegacySearchWarning = oldOption
  ), add = TRUE)
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


#' setResNet
#'
#' @description
#' Creates settings for a ResNet model
#'
#' @details
#' Model architecture from by https://arxiv.org/abs/2106.11959
#'
#'
#' @param numLayers         Number of layers in network, default: 1:16
#' @param sizeHidden        Amount of neurons in each default layer, default:
#' 2^(6:10) (64 to 1024)
#' @param hiddenFactor      How much to grow the amount of neurons in each
#' ResLayer, default: 1:4
#' @param residualDropout   How much dropout to apply after last linear layer
#' in ResLayer, default: seq(0, 0.3, 0.05)
#' @param hiddenDropout     How much dropout to apply after first linear layer
#' in ResLayer, default: seq(0, 0.3, 0.05)
#' @param sizeEmbedding     Size of embedding layer, default: 2^(6:9)
#' '(64 to 512)
#' @param estimatorSettings created with ```setEstimator```
#' @param hyperParamSearch Deprecated. Use PLP `hyperparameterSettings`
#' instead.
#' @param randomSample Deprecated. Use PLP `hyperparameterSettings` instead.
#' @param randomSampleSeed Deprecated. Use PLP `hyperparameterSettings`
#' instead.
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
  legacySearchExplicit <- !missing(hyperParamSearch) ||
    !missing(randomSample) ||
    !missing(randomSampleSeed)
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

  results <- createDeepModelSettings(
    paramDefinition = paramGrid,
    estimatorSettings = estimatorSettings,
    modelParamNames = c(
      "numLayers", "sizeHidden", "hiddenFactor",
      "residualDropout", "hiddenDropout", "sizeEmbedding"
    ),
    modelType = "ResNet",
    hyperParamSearch = hyperParamSearch,
    randomSample = randomSample,
    randomSampleSeed = randomSampleSeed,
    legacySearchExplicit = legacySearchExplicit
  )

  return(results)
}
