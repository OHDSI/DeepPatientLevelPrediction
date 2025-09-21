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
#' @param attnImplementation      attention implementation to use, either
#' 'sdpa' (scaled dot product attention from pytorch) or 'flash" (flash attention v2
#' from `flash-attn` package). Flash requires a new GPU with compute capability 8.0
#' or higher, bfloat16 and the `flash-attn` package to be installed.
#' If you chose flash attention you need to force the which python environment 
#' is and need to make sure it has all required packages installed.
#' @param temporal                Whether to use a transformer with temporal data
#' @param temporalSettings        settings for the temporal transformer. Which include
#'   - `positionalEncoding`: Positional encoding to use, either a character
#'     or a list with name and settings, default 'SinusoidalPE' with dropout 0.1
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
                           attnImplementation = "sdpa",
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
  checkIsClass(attnImplementation, "character")
  if (!(attnImplementation %in% c("sdpa", "flash"))) {
    stop(paste(
      "attnImplementation must be either 'sdpa' or 'flash'. You provided: ",
      attnImplementation
    ))
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
    ffnDropout = ffnDropout,
    attnImplementation = attnImplementation
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
  if (attnImplementation == "flash") {
    envCheck <- flashEnvCheck(estimatorSettings)
    if (!envCheck$ok) {
      stop(
        formatFlashErrors(
          "FlashAttention-2 environment validation failed for the following reasons:", 
          envCheck$reasons
        ),
        call. = FALSE
      )
    }
    paramChecks <- lapply(param, function(x) {
      flashParamCheck(
        paramCombo = x,
        temporal = temporal,
        temporalSettings = temporalSettings,
        envInfo = envCheck
      )
    }
    )

    badIdx <- which(!vapply(paramChecks, function(z) z$ok, logical(1)))
    if (length(badIdx) > 0) {
      lines <- c("FlashAttention-2 is not supported for the following hyperparameter combinations:")
      for (i in badIdx) {
        x <- param[[i]]
        peName <- if (!is.null(x$positionalEncoding) && is.list(x$positionalEncoding))
          x$positionalEncoding$name %||% NA_character_ else NA_character_
        lines <- c(
          lines,
          paste0("  combo #", i,
                 " (dimToken=", x$dimToken,
                 ", numHeads=", x$numHeads,
                 if (!is.na(peName)) paste0(", positionalEncoding='", peName, "'") else "",
                 "):"),
          paste0("     - ", paramChecks[[i]]$reasons)
        )
      }
    stop(paste0(lines, collapse = "\n"), call. = FALSE)
    }
  }


  results <- list(
    fitFunction = "DeepPatientLevelPrediction::fitEstimator",
    param = param,
    estimatorSettings = estimatorSettings,
    saveType = "file",
    modelParamNames = c(
      "numBlocks", "dimToken", "dimOut", "numHeads",
      "attDropout", "ffnDropout", "dimHidden", "attnImplementation"
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

# Check if flash attention v2 can be used
flashEnvCheck <- function(estimatorSettings) {
  reasons <- character(0)
  info <- list()

  # FlashAttention-2 effectively targets Linux
  sysName <- getSysName()
  if (sysName != "linux") {
    reasons <- c(
      reasons, paste0("FlashAttention 2 requires Linux; detected '", 
      Sys.info()[["sysname"]], "'."))
  }

  device <- estimatorSettings$device %||% "cpu"
  if (!identical(device, "cuda")) {
    reasons <- c(reasons, "estimatorSettings$device must be 'cuda' to use FlashAttention 2.")
  }

  pt <- getTorch()
  if (!isTRUE(pt$cuda$is_available())) {
    reasons <- c(reasons, "torch.cuda.is_available() is FALSE; a CUDA-enabled PyTorch is required.")
  }

  torchCudaVersion <- tryCatch(pt$version$cuda, error = function(e) NULL)
  if (is.null(torchCudaVersion) || is.na(torchCudaVersion)) {
    reasons <- c(reasons, "torch.version.cuda is NULL; your PyTorch does not have a CUDA runtime.")
  } else {
    info$torchCudaVersion <- torchCudaVersion
  }

  if (!pyModuleAvailable("flash_attn.flash_attn_interface")) {
    reasons <- c(
      reasons,
      paste0(
        "flash-attn (v2) is not importable in the Python env used by reticulate. ",
        "Point reticulate to a Python env where flash-attn is preinstalled (built from source). ",
        "Example before loading the package:\n",
        "  Sys.setenv(RETICULATE_PYTHON='/path/to/python')\n",
        "Then restart R and library(DeepPatientLevelPrediction)."
      )
    )
  } else {
    fa <- reticulate::import("flash_attn", delay_load = TRUE)
    info$flashAttnVersion <- tryCatch(fa$`__version__`, error = function(e) NA_character_)
  }

  cap <- tryCatch(pt$cuda$get_device_capability(), error = function(e) list(0L, 0L))
  major <- as.integer(cap[[1]])
  minor <- as.integer(cap[[2]])
  info$computeCapability <- paste0(major, ".", minor)
  info$deviceName <- tryCatch(pt$cuda$get_device_name(), error = function(e) NA_character_)
  if (is.na(major) || major < 8L) {
    reasons <- c(
      reasons, 
      "FlashAttention 2 requires NVIDIA Ampere/Ada/Hopper or newer (compute capability >= 8.0).")
  }

  bf16Supported <- isTRUE(tryCatch(pt$cuda$is_bf16_supported(), error = function(e) FALSE))
  if (!bf16Supported) {
    reasons <- c(
      reasons, 
      "BF16 is not supported on this GPU/driver/torch build (torch.cuda.is_bf16_supported() is FALSE).")
  }
  precisionVal <- estimatorSettings$precision

  if (!identical(precisionVal, "bfloat16")) {
    reasons <- c(reasons, 
      paste0(
        "precision must be 'bfloat16' when using FlashAttention 2; got '", 
        precisionVal, "'."
      )
    )
  }
  list(ok = length(reasons) == 0, reasons = reasons, info = info)
}

# Check if the paramCombo is compatible with flash attention v2
flashParamCheck <- function(paramCombo, temporal, temporalSettings, envInfo) {
  reasons <- character(0)

  if ((paramCombo$dimToken %% paramCombo$numHeads) != 0) {
    reasons <- c(
      reasons,
      paste0(
        "dimToken (", paramCombo$dimToken, ") must be divisible by numHeads (", 
        paramCombo$numHeads, ")."
      )
    )
  } else {
    headDim <- as.integer(paramCombo$dimToken / paramCombo$numHeads)
    if ((headDim %% 8L) != 0L) {
      reasons <- c(
        reasons, 
        paste0("headDim must be a multiple of 8 for flash-attn; got ", headDim, ".")
      )
    }
    if (headDim > 256L) {
      reasons <- c(
        reasons, 
        paste0("headDim must be <= 256 for flash-attn; got ", headDim, ".")
      )
    }
  }

  peName <- NULL
  if (temporal) {
    if (!is.null(paramCombo$positionalEncoding) && is.list(paramCombo$positionalEncoding)) {
      peName <- paramCombo$positionalEncoding$name %||% NULL
    } else if (!is.null(temporalSettings$positionalEncoding) && 
      is.list(temporalSettings$positionalEncoding)) {
      peName <- temporalSettings$positionalEncoding$name %||% NULL
    }
  }

  disallowed <- c(
    "RelativePE", 
    "TemporalPE", 
    "TUPE",
    "EfficientRPE",
    "ALiBiPE"
  )

  allowed <- c(
    "NoPositionalEncoding", 
    "SinusoidalPE", 
    "LearnablePE",
    "TapePE",
    "RotaryPE", 
    "StochasticConvPE",
    "HybridRoPEConvPE"
  )

  if (!is.null(peName) && (peName %in% disallowed)) {
    reasons <- c(
      reasons,
      paste0("positionalEncoding '", peName, 
        "' injects additive attention bias/scores and is not supported by flash-attn v2.")
    )
  }
  if (!is.null(peName) && !(peName %in% c(disallowed, allowed))) {
    reasons <- c(
      reasons,
      paste0("Unknown positionalEncoding '", peName, 
        "'. Ensure it does not add attention bias/scores for flash-attn v2.")
    )
  }

  list(ok = length(reasons) == 0, reasons = reasons)
}

formatFlashErrors <- function(header, reasons) {
  if (length(reasons) == 0) return("")
  paste0(header, "\n", paste0("  - ", reasons, collapse = "\n"))
}

# thin wrappers for easier mocking in tests
pyModuleAvailable <- function(mod) {
  reticulate::py_module_available(mod)
}

getTorch <- function() {
  reticulate::import("torch", delay_load = TRUE)
}

getSysName <- function() {
  tolower(Sys.info()[["sysname"]])
}
