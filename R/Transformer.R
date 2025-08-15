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
                                    setEstimator(
                                      learningRate = "auto",
                                      weightDecay = 1e-4,
                                      batchSize = 512,
                                      epochs = 10,
                                      seed = NULL,
                                      device = "cpu"
                                    )) {
  transformerSettings <- setTransformer(
    numBlocks = 3,
    dimToken = 192,
    dimOut = 1,
    numHeads = 8,
    attDropout = 0.2,
    ffnDropout = 0.1,
    dimHidden = 256,
    estimatorSettings = estimatorSettings,
    hyperParamSearch = "random",
    randomSample = 1
  )
  attr(transformerSettings, "settings")$name <- "defaultTransformer"
  return(transformerSettings)
}

#' create settings for training a transformer
#'
#' @description A transformer model
#' @details The non-temporal transformer is from https://arxiv.org/abs/2106.11959
#'
#' @param numBlocks               number of transformer blocks
#' @param dimToken                dimension of each token (embedding size)
#' @param dimOut                  dimension of output, usually 1 for binary
#' problems
#' @param numHeads                number of attention heads
#' @param attDropout              dropout to use on attentions
#' @param ffnDropout              dropout to use in feedforward block
#' @param dimHidden               dimension of the feedworward block
#' @param dimHiddenRatio          dimension of the feedforward block as a ratio
#' of dimToken (embedding size)
#' @param temporal                Whether to use a transformer with temporal data
#' @param temporalSettings        settings for the temporal transformer. Which include
#'   - `maxSequenceLength`: Maximum sequence length, sequences longer than This
#'     will be truncated and/or padded to this length either a number or 'max' for the Maximum
#'   - `truncation`: Truncation method, only 'tail' is supported
#'   - `timeTokens`: Whether to use time tokens, default TRUE
#' @param estimatorSettings       created with `setEstimator`
#' @param hyperParamSearch        what kind of hyperparameter search to do,
#' default 'random'
#' @param randomSample            How many samples to use in hyperparameter
#' search if random
#' @param randomSampleSeed        Random seed to sample hyperparameter
#' combinations
#' @return list of settings for the transformer model
#'
#' @export
setTransformer <- function(numBlocks = 3,
                           dimToken = 192,
                           dimOut = 1,
                           numHeads = 8,
                           attDropout = 0.2,
                           ffnDropout = 0.1,
                           dimHidden = 256,
                           dimHiddenRatio = NULL,
                           temporal = FALSE,
                           temporalSettings = list(
                             positionalEncoding = list(
                               name = "SinusoidalPE",
                               dropout = 0.1
                             ),
                             maxSequenceLength = 256,
                             truncation = "tail",
                             timeTokens = TRUE
                           ),
                           estimatorSettings = setEstimator(
                             weightDecay = 1e-6,
                             batchSize = 1024,
                             epochs = 10,
                             seed = NULL
                           ),
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

  checkIsClass(dimHidden, c("integer", "numeric", "NULL"))
  if (!is.null(dimHidden)) {
    checkHigherEqual(dimHidden, 1)
  }
  checkIsClass(temporal, "logical")

  checkIsClass(dimHiddenRatio, c("numeric", "NULL"))
  if (!is.null(dimHiddenRatio)) {
    checkHigher(dimHiddenRatio, 0)
  }

  checkIsClass(hyperParamSearch, "character")

  checkIsClass(randomSample, c("numeric", "integer"))
  checkHigherEqual(randomSample, 1)

  checkIsClass(randomSampleSeed, c("numeric", "integer", "NULL"))

  if (any(with(
    expand.grid(dimToken = dimToken, numHeads = numHeads),
    dimToken %% numHeads != 0
  ))) {
    stop(paste(
      "dimToken needs to divisible by numHeads. dimToken =", dimToken,
      "is not divisible by numHeads =", numHeads
    ))
  }

  if (is.null(dimHidden) && is.null(dimHiddenRatio) ||
    !is.null(dimHidden) && !is.null(dimHiddenRatio)) {
    stop(paste(
      "dimHidden and dimHiddenRatio cannot be both set or both NULL"
    ))
  } else if (!is.null(dimHiddenRatio)) {
    dimHidden <- dimHiddenRatio
  }

  checkIsClass(
    temporalSettings$maxSequenceLength,
    c("integer", "numeric", "character")
  )
  if (!inherits(temporalSettings$maxSequenceLength, "character")) {
    checkHigherEqual(temporalSettings$maxSequenceLength, 1)
  } else if (temporalSettings$maxSequenceLength != "max") {
    stop(paste(
      "maxSequenceLength must be either 'max' or a positive integer. maxSequenceLength =",
      temporalSettings$maxSequenceLength
    ))
  }
  if (inherits(temporalSettings$maxSequenceLength, "numeric")) {
    temporalSettings$maxSequenceLength <-
      as.integer(round(temporalSettings$maxSequenceLength))
  }
  checkIsClass(temporalSettings$truncation, "character")
  if (temporalSettings$truncation != "tail") {
    stop(paste(
      "Only truncation method 'tail' is supported. truncation =",
      temporalSettings$truncation
    ))
  }
  checkIsClass(temporalSettings$positionalEncoding, c("character", "list"))
  if (inherits(temporalSettings$positionalEncoding, "character")) {
    temporalSettings$positionalEncoding <- list(name = temporalSettings$positionalEncoding)
  }

  paramGrid <- list(
    numBlocks = numBlocks,
    dimToken = dimToken,
    dimOut = dimOut,
    numHeads = numHeads,
    dimHidden = dimHidden,
    attDropout = attDropout,
    ffnDropout = ffnDropout
  )
  if (temporal) {
    paramGrid[["positionalEncoding"]] <- 
      expandComponentGrid(temporalSettings$positionalEncoding)
  }

  paramGrid <- c(paramGrid, estimatorSettings$paramsToTune)

  param <- PatientLevelPrediction::listCartesian(paramGrid)

  if (!is.null(dimHiddenRatio)) {
    param <- lapply(param, function(x) {
      x$dimHidden <- round(x$dimToken * x$dimHidden, digits = 0)
      return(x)
    })
  }

  if (hyperParamSearch == "random" && randomSample > length(param)) {
    stop(paste(
      "\n Chosen amount of randomSamples is higher than the amount of
               possible hyperparameter combinations.", "\n randomSample:",
      randomSample, "\n Possible hyperparameter combinations:",
      length(param), "\n Please lower the amount of randomSample"
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
    modelParamNames = c(
      "numBlocks", "dimToken", "dimOut", "numHeads",
      "attDropout", "ffnDropout", "dimHidden"
    ),
    modelType = "Transformer"
  )
  if (temporal) {
    attr(results$param, "temporalModel") <- TRUE
    attr(results$param, "temporalSettings") <- temporalSettings
    results$modelParamNames <- c(results$modelParamNames, "positionalEncoding")
  }
  attr(results$param, "settings")$modelType <- results$modelType
  class(results) <- "modelSettings"
  return(results)
}
