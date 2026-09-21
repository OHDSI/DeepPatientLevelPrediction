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
#' Creates settings for the package's default non-temporal transformer model.
#'
#' @details The architecture and default hyperparameters are based on
#' [Gorishniy et al. (2021)](https://arxiv.org/abs/2106.11959).
#'
#' @param estimatorSettings Estimator settings created by [setEstimator()].
#'
#' @return A `modelSettings` object for use with `PatientLevelPrediction`.
#'
#' @examples
#' transformerSettings <- setDefaultTransformer()
#' transformerSettings$param[[1]]$numBlocks
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

#' Create Transformer Settings
#'
#' Creates model and hyperparameter-search settings for either a non-temporal
#' or temporal transformer.
#'
#' @details The non-temporal architecture is based on
#' [Gorishniy et al. (2021)](https://arxiv.org/abs/2106.11959). For temporal
#' data, positional encoding can be configured through `temporalSettings`.
#'
#' @param numBlocks Number of transformer blocks.
#' @param dimToken Token dimension, which is also the embedding dimension.
#' @param dimOut Output dimension, usually one for binary prediction.
#' @param numHeads Number of attention heads.
#' @param attDropout Attention-dropout probability.
#' @param ffnDropout Feed-forward-network dropout probability.
#' @param dimHidden Hidden dimension of the feed-forward network.
#' @param dimHiddenRatio Feed-forward hidden dimension as a ratio of
#'   `dimToken`. Exactly one of `dimHidden` and `dimHiddenRatio` must be `NULL`.
#' @param temporal Whether to configure a transformer for temporal covariates.
#' @param temporalSettings A list with `positionalEncoding`,
#'   `maxSequenceLength`, `truncation`, and `timeTokens`. The only supported
#'   truncation strategy is `"tail"`.
#' @param estimatorSettings Estimator settings created by [setEstimator()].
#' @param hyperParamSearch Hyperparameter-search strategy, either `"random"`
#'   or `"grid"`.
#' @param randomSample Number of combinations sampled when
#'   `hyperParamSearch = "random"`.
#' @param randomSampleSeed Random seed used when sampling combinations.
#'
#' @return A `modelSettings` object. Temporal settings are stored as attributes
#'   on its parameter grid.
#'
#' @examples
#' transformerSettings <- setTransformer(
#'   numBlocks = 2,
#'   dimToken = 64,
#'   numHeads = 4,
#'   dimHidden = 128
#' )
#'
#' temporalSettings <- setTransformer(
#'   numBlocks = 1,
#'   dimToken = 32,
#'   numHeads = 4,
#'   dimHidden = 64,
#'   temporal = TRUE,
#'   temporalSettings = list(
#'     positionalEncoding = "SinusoidalPE",
#'     maxSequenceLength = 128,
#'     truncation = "tail",
#'     timeTokens = TRUE
#'   )
#' )
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
  defaultTemporalSettings <- list(
    positionalEncoding = list(
      name = "SinusoidalPE",
      dropout = 0.1
    ),
    maxSequenceLength = 256,
    truncation = "tail",
    timeTokens = FALSE
  )
  temporalSettings <- keepDefaults(
    defaultTemporalSettings,
    temporalSettings
  )

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
  checkIsClass(temporalSettings$positionalEncoding, c("character", "list", "NULL"))
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
    if (!is.null(temporalSettings$positionalEncoding)) {
      paramGrid[["positionalEncoding"]] <- 
        expandComponentGrid(temporalSettings$positionalEncoding)
    }
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
    if (!is.null(temporalSettings$positionalEncoding)) {
      results$modelParamNames <- c(results$modelParamNames, "positionalEncoding")
    }
  }
  attr(results$param, "settings")$modelType <- results$modelType
  class(results) <- "modelSettings"
  return(results)
}
