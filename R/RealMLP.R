# @file: RealMLP.R

#' Create RealMLP Settings
#'
#' Creates model and hyperparameter-search settings for RealMLP binary
#' classification.
#'
#' @details The implementation is based on
#' [Holzmuller et al. (2024)](https://arxiv.org/abs/2407.04491).
#' Preprocessing, including robust scaling and clipping, is handled upstream.
#'
#' @param numLayers  hidden layers (default 3)
#' @param sizeHidden hidden width (default 256)
#' @param dropout    base dropout p (default 0.15; scheduled with flat_cos)
#' @param sizeEmbedding embedding dim for compatibility with existing Embedding (default 64)
#' @param labelSmoothing epsilon for label smoothing (default 0.0 for AUROC mode)
#' @param numericEmbeddingMode numeric token embedding mode: "scale", "pl", or "pbld"
#' @param numericNumFrequencies periodic frequency count for PL/PBLD modes
#' @param numericPeriodicInitStd std used to initialize periodic frequencies
#' @param numericPbldHiddenDim hidden width for PBLD per-feature block
#' @param numericPbldEmbeddingDim low-dimensional PBLD output width before projection
#' @param dataDependentInitMode data-dependent init mode: "paper_lsuv" or "current"
#' @param dataDependentInitTargetVar target pre-activation variance per neuron
#' @param dataDependentInitMaxRows max sampled rows for init statistics (0 means all sampled rows)
#' @param dataDependentInitGainClip optional clip on LSUV gain multipliers (>1 enables clipping)
#' @param dataDependentInitBiasRefitSteps bias recenter iterations during data-dependent init
#' @param scalingLrMult LR multiplier for scaling parameters (default 6.0)
#' @param biasLrMult LR multiplier for bias parameters (default 0.1)
#' @param actLrMult LR multiplier for parametric activation parameters (default 0.1)
#' @param embeddingLrMult LR multiplier for embedding parameters (default 0.1)
#' @param paperMode if TRUE, enforce paper-aligned defaults where possible
#' @param tokenAggregation token aggregation mode: "auto", "mean", "sum", "sum_len_norm"
#' @param featureScaleMode feature scale mode: "auto", "scalar", "vector"
#' @param estimatorSettings Estimator settings created by [setEstimator()].
#'   The default trains as in the RealMLP paper: AdamW with learning rate
#'   2e-3 and weight decay 0.02, batches of 256 and 256 epochs, keeping the
#'   epoch with the lowest validation loss. RealMLP follows its own
#'   learning-rate schedule, so a `scheduler` is ignored with a warning, and
#'   its own loss, so `criterion` is not used. Early stopping is possible, but
#'   the paper trains for all epochs.
#' @param hyperParamSearch Which kind of hyperparameter search to use: random
#' sampling or exhaustive grid search. Default: "grid"
#' @param randomSample How many random samples from hyperparameter space to use
#' when `hyperParamSearch = "random"`
#' @param randomSampleSeed Random seed to sample hyperparameter combinations
#' @param device Deprecated. Use `estimatorSettings = setEstimator(device = )`.
#'
#' @return A `modelSettings` object for use with `PatientLevelPrediction`.
#'
#' @examples
#' realMlpSettings <- setRealMLP(
#'   sizeHidden = c(128, 256),
#'   dropout = c(0.1, 0.2),
#'   hyperParamSearch = "random",
#'   randomSample = 2,
#'   randomSampleSeed = 42
#' )
#' length(realMlpSettings$param)
#' @export
setRealMLP <- function(
  numLayers = 3L,
  sizeHidden = 256L,
  dropout = 0.15,
  sizeEmbedding = 64L,
  labelSmoothing = 0.0,
  numericEmbeddingMode = "scale",
  numericNumFrequencies = 8L,
  numericPeriodicInitStd = 0.1,
  numericPbldHiddenDim = 16L,
  numericPbldEmbeddingDim = 4L,
  dataDependentInitMode = "paper_lsuv",
  dataDependentInitTargetVar = 1.0,
  dataDependentInitMaxRows = 65536L,
  dataDependentInitGainClip = 10.0,
  dataDependentInitBiasRefitSteps = 2L,
  scalingLrMult = 6.0,
  biasLrMult = 0.1,
  actLrMult = 0.1,
  embeddingLrMult = 0.1,
  paperMode = TRUE,
  tokenAggregation = "auto",
  featureScaleMode = "auto",
  estimatorSettings = setEstimator(
    learningRate = 2e-3,
    weightDecay = 0.02,
    batchSize = 256L,
    epochs = 256L,
    scheduler = NULL,
    earlyStopping = NULL,
    metric = "loss"
  ),
  hyperParamSearch = "grid",
  randomSample = 100,
  randomSampleSeed = NULL,
  device = NULL
) {
  checkIsClass(numLayers, c("integer", "numeric"))
  checkHigherEqual(numLayers, 1)
  checkIsClass(sizeHidden, c("integer", "numeric"))
  checkHigherEqual(sizeHidden, 1)
  checkIsClass(dropout, "numeric")
  checkHigherEqual(dropout, 0)
  checkLower(dropout, 1)
  checkIsClass(sizeEmbedding, c("integer", "numeric"))
  checkHigherEqual(sizeEmbedding, 1)
  checkIsClass(labelSmoothing, "numeric")
  checkHigherEqual(labelSmoothing, 0)
  if (any(labelSmoothing > 1)) {
    stop("labelSmoothing needs to be <= 1")
  }
  checkIsClass(numericEmbeddingMode, "character")
  if (!all(numericEmbeddingMode %in% c("scale", "pl", "pbld"))) {
    stop("numericEmbeddingMode has incorrect value")
  }
  checkIsClass(numericNumFrequencies, c("integer", "numeric"))
  checkHigherEqual(numericNumFrequencies, 1)
  checkIsClass(numericPeriodicInitStd, "numeric")
  checkHigherEqual(numericPeriodicInitStd, 0)
  checkIsClass(numericPbldHiddenDim, c("integer", "numeric"))
  checkHigherEqual(numericPbldHiddenDim, 1)
  checkIsClass(numericPbldEmbeddingDim, c("integer", "numeric"))
  checkHigherEqual(numericPbldEmbeddingDim, 1)
  checkIsClass(dataDependentInitMode, "character")
  if (!all(dataDependentInitMode %in% c("paper_lsuv", "current"))) {
    stop("dataDependentInitMode has incorrect value")
  }
  checkIsClass(dataDependentInitTargetVar, "numeric")
  checkHigher(dataDependentInitTargetVar, 0)
  checkIsClass(dataDependentInitMaxRows, c("integer", "numeric"))
  checkHigherEqual(dataDependentInitMaxRows, 0)
  checkIsClass(dataDependentInitGainClip, "numeric")
  checkHigherEqual(dataDependentInitGainClip, 0)
  checkIsClass(dataDependentInitBiasRefitSteps, c("integer", "numeric"))
  checkHigherEqual(dataDependentInitBiasRefitSteps, 1)
  checkIsClass(scalingLrMult, "numeric")
  checkHigherEqual(scalingLrMult, 0)
  checkIsClass(biasLrMult, "numeric")
  checkHigherEqual(biasLrMult, 0)
  checkIsClass(actLrMult, "numeric")
  checkHigherEqual(actLrMult, 0)
  checkIsClass(embeddingLrMult, "numeric")
  checkHigherEqual(embeddingLrMult, 0)
  checkIsClass(paperMode, "logical")
  checkIsClass(tokenAggregation, "character")
  if (!all(tokenAggregation %in% c("auto", "mean", "sum", "sum_len_norm"))) {
    stop("tokenAggregation has incorrect value")
  }
  checkIsClass(featureScaleMode, "character")
  if (!all(featureScaleMode %in% c("auto", "scalar", "vector"))) {
    stop("featureScaleMode has incorrect value")
  }
  if (!is.null(device)) {
    warning(
      "`device` is deprecated. Use `estimatorSettings = setEstimator(device = )`.",
      call. = FALSE
    )
    checkIsClass(device, c("character", "function"))
    estimatorSettings$device <- device
    if (is.function(device)) {
      class(estimatorSettings$device) <- c("delayed", class(device))
    }
  }
  if (!isDelayedNull(estimatorSettings$scheduler)) {
    warning(
      "RealMLP follows its own learning-rate schedule, so the scheduler in ",
      "`estimatorSettings` is ignored.",
      call. = FALSE
    )
    estimatorSettings$scheduler <- delaySetting(rlang::quo(NULL))
  }

  checkIsClass(hyperParamSearch, "character")
  checkInStringVector(hyperParamSearch, c("random", "grid"))

  checkIsClass(randomSample, c("numeric", "integer"))
  checkHigherEqual(randomSample, 1)

  checkIsClass(randomSampleSeed, c("numeric", "integer", "NULL"))

  # RealMLP's own training settings are model parameters, which RealMLP reads
  paramGrid <- list(
    numLayers = as.integer(numLayers),
    sizeHidden = as.integer(sizeHidden),
    dropout = dropout,
    sizeEmbedding = as.integer(sizeEmbedding),
    labelSmoothing = labelSmoothing,
    numericEmbeddingMode = numericEmbeddingMode,
    numericNumFrequencies = as.integer(numericNumFrequencies),
    numericPeriodicInitStd = numericPeriodicInitStd,
    numericPbldHiddenDim = as.integer(numericPbldHiddenDim),
    numericPbldEmbeddingDim = as.integer(numericPbldEmbeddingDim),
    dataDependentInitMode = dataDependentInitMode,
    dataDependentInitTargetVar = dataDependentInitTargetVar,
    dataDependentInitMaxRows = as.integer(dataDependentInitMaxRows),
    dataDependentInitGainClip = dataDependentInitGainClip,
    dataDependentInitBiasRefitSteps = as.integer(dataDependentInitBiasRefitSteps),
    scalingLrMult = scalingLrMult,
    biasLrMult = biasLrMult,
    actLrMult = actLrMult,
    embeddingLrMult = embeddingLrMult,
    paperMode = paperMode,
    tokenAggregation = tokenAggregation,
    featureScaleMode = featureScaleMode
  )
  paramGrid <- c(paramGrid, estimatorSettings$paramsToTune)

  param <- PatientLevelPrediction::listCartesian(paramGrid)
  param <- lapply(param, function(x) {
    if (x$tokenAggregation == "auto") {
      x$tokenAggregation <- if (isTRUE(x$paperMode)) "sum" else "mean"
    }
    # RealMLP() replaces mean by sum in paper mode, which models saved
    # before this check rely on
    if (isTRUE(x$paperMode) && x$tokenAggregation == "mean") {
      stop(
        "tokenAggregation = \"mean\" can't be combined with paperMode = TRUE, ",
        "which sums the tokens. Use \"auto\", \"sum\" or \"sum_len_norm\"."
      )
    }
    if (x$featureScaleMode == "auto") {
      x$featureScaleMode <- "scalar"
    }
    x
  })

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
    modelParamNames = c(
      "numLayers", "sizeHidden", "dropout", "sizeEmbedding",
      "labelSmoothing", "numericEmbeddingMode", "numericNumFrequencies",
      "numericPeriodicInitStd", "numericPbldHiddenDim",
      "numericPbldEmbeddingDim", "dataDependentInitMode",
      "dataDependentInitTargetVar", "dataDependentInitMaxRows",
      "dataDependentInitGainClip", "dataDependentInitBiasRefitSteps",
      "scalingLrMult", "biasLrMult", "actLrMult", "embeddingLrMult",
      "paperMode", "tokenAggregation", "featureScaleMode"
    ),
    modelType = "RealMLP"
  )
  attr(results$param, "settings")$modelType <- results$modelType
  class(results) <- "modelSettings"
  results
}
