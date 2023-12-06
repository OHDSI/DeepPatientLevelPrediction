# @file Transformer.R
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

#' Create default settings for a non-temporal transformer
#'
#' @description A transformer model with default hyperparameters
#' @details from https://arxiv.org/abs/2106.11959
#' Default hyperparameters from paper
#' @param estimatorSettings created with `setEstimator`
#'
#' @export
setDefaultTransformer <- function(estimatorSettings =
    setEstimator(learningRate = "auto",
                 weightDecay = 1e-4,
                 batchSize = 512,
                 epochs = 10,
                 seed = NULL,
                 device = "cpu")
) {
  transformerSettings <- setTransformer(numBlocks = 3,
                                        dimToken = 192,
                                        dimOut = 1,
                                        numHeads = 8,
                                        attDropout = 0.2,
                                        ffnDropout = 0.1,
                                        resDropout = 0.0,
                                        dimHidden = 256,
                                        estimatorSettings = estimatorSettings,
                                        hyperParamSearch = "random",
                                        randomSample = 1)
  attr(transformerSettings, "settings")$name <- "defaultTransformer"
  return(transformerSettings)
}

#' create settings for training a non-temporal transformer
#'
#' @description A transformer model
#' @details from https://arxiv.org/abs/2106.11959
#'
#' @param numBlocks               number of transformer blocks
#' @param dimToken                dimension of each token (embedding size)
#' @param dimOut                  dimension of output, usually 1 for binary
#' problems
#' @param numHeads                number of attention heads
#' @param attDropout              dropout to use on attentions
#' @param ffnDropout              dropout to use in feedforward block
#' @param resDropout              dropout to use in residual connections
#' @param dimHidden               dimension of the feedworward block
#' @param dimHiddenRatio          dimension of the feedforward block as a ratio
#' of dimToken (embedding size)
#' @param estimatorSettings       created with `setEstimator`
#' @param hyperParamSearch        what kind of hyperparameter search to do,
#' default 'random'
#' @param randomSample            How many samples to use in hyperparameter
#' search if random
#' @param randomSampleSeed        Random seed to sample hyperparameter
#' combinations
#'
#' @export
setTransformer <- function(numBlocks = 3,
                           dimToken = 96,
                           dimOut = 1,
                           numHeads = 8,
                           attDropout = 0.25,
                           ffnDropout = 0.25,
                           resDropout = 0,
                           dimHidden = 512,
                           dimHiddenRatio = NULL,
                           estimatorSettings = setEstimator(weightDecay = 1e-6,
                                                            batchSize = 1024,
                                                            epochs = 10,
                                                            seed = NULL),
                           hyperParamSearch = "random",
                           randomSample = 1,
                           randomSampleSeed = NULL) {

  checkIsClass(numBlocks, c("integer", "numeric"))
  checkHigherEqual(numBlocks, 1)

  checkIsClass(dimToken, c("integer", "numeric"))
  checkHigherEqual(dimToken, 1)

  checkIsClass(dimOut, c("integer", "numeric"))
  checkHigherEqual(dimOut, 1)

  checkIsClass(numHeads, c("integer", "numeric"))
  checkHigherEqual(numHeads, 1)

  checkIsClass(attDropout, c("numeric"))
  checkHigherEqual(attDropout, 0)

  checkIsClass(ffnDropout, c("numeric"))
  checkHigherEqual(ffnDropout, 0)

  checkIsClass(resDropout, c("numeric"))
  checkHigherEqual(resDropout, 0)

  checkIsClass(dimHidden, c("integer", "numeric", "NULL"))
  if (!is.null(dimHidden)) {
    checkHigherEqual(dimHidden, 1)
  }

  checkIsClass(dimHiddenRatio, c("numeric", "NULL"))
  if (!is.null(dimHiddenRatio)) {
    checkHigher(dimHiddenRatio, 0)
  }

  checkIsClass(hyperParamSearch, "character")

  checkIsClass(randomSample, c("numeric", "integer"))
  checkHigherEqual(randomSample, 1)

  checkIsClass(randomSampleSeed, c("numeric", "integer", "NULL"))

  if (any(with(expand.grid(dimToken = dimToken, numHeads = numHeads),
               dimToken %% numHeads != 0))) {
    stop(paste(
      "dimToken needs to divisible by numHeads. dimToken =", dimToken,
      "is not divisible by numHeads =", numHeads
    ))
  }

  if (is.null(dimHidden) && is.null(dimHiddenRatio)
      || !is.null(dimHidden) && !is.null(dimHiddenRatio)) {
    stop(paste(
      "dimHidden and dimHiddenRatio cannot be both set or both NULL"
    ))
  } else {
    if (!is.null(dimHiddenRatio)) {
      dimHidden <- dimHiddenRatio
    }
  }

  paramGrid <- list(
    numBlocks = numBlocks,
    dimToken = dimToken,
    dimOut = dimOut,
    numHeads = numHeads,
    dimHidden = dimHidden,
    attDropout = attDropout,
    ffnDropout = ffnDropout,
    resDropout = resDropout
  )

  paramGrid <- c(paramGrid, estimatorSettings$paramsToTune)

  param <- PatientLevelPrediction::listCartesian(paramGrid)

  if (!is.null(dimHiddenRatio)) {
    param <- lapply(param, function(x) {
      x$dimHidden <- round(x$dimToken * x$dimHidden, digits = 0)
      return(x)
    })
  }

  if (hyperParamSearch == "random" && randomSample > length(param)) {
    stop(paste("\n Chosen amount of randomSamples is higher than the amount of
               possible hyperparameter combinations.", "\n randomSample:",
               randomSample, "\n Possible hyperparameter combinations:",
               length(param), "\n Please lower the amount of randomSample"))
  }

  if (hyperParamSearch == "random") {
    suppressWarnings(withr::with_seed(randomSampleSeed,
                                      {param <- param[sample(length(param),
                                                             randomSample)]}))
  }
  estimatorSettings$modelType <- "Transformer"
  attr(param, "settings")$modelType <- estimatorSettings$modelType
  results <- list(
    fitFunction = "fitEstimator",
    param = param,
    estimatorSettings = estimatorSettings,
    saveType = "file",
    modelParamNames = c(
      "numBlocks", "dimToken", "dimOut", "numHeads",
      "attDropout", "ffnDropout", "resDropout", "dimHidden"
    )
  )

  class(results) <- "modelSettings"
  return(results)
}
