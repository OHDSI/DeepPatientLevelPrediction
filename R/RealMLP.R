# @file: RealMLP.R

#' Create RealMLP Settings
#'
#' Create settings for the RealMLP model (binary classification).
#' Preprocessing (robust scaling + clipping) is assumed to be handled upstream.
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
#' @param device     "cpu" or "cuda" (default "cpu")
#' @param hyperParamSearch Deprecated. Use PLP `hyperparameterSettings`
#' instead.
#' @param randomSample Deprecated. Use PLP `hyperparameterSettings` instead.
#' @param randomSampleSeed Deprecated. Use PLP `hyperparameterSettings`
#' instead.
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
  device = "cpu",
  hyperParamSearch = "grid",
  randomSample = 100,
  randomSampleSeed = NULL
) {
  legacySearchExplicit <- !missing(hyperParamSearch) ||
    !missing(randomSample) ||
    !missing(randomSampleSeed)
  checkIsClass(numLayers, c("integer", "numeric"))
  checkHigherEqual(numLayers, 1)
  checkIsClass(sizeHidden, c("integer", "numeric"))
  checkHigherEqual(sizeHidden, 1)
  checkIsClass(dropout, "numeric")
  checkHigherEqual(dropout, 0)
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
  checkIsClass(device, c("character", "function"))

  checkIsClass(hyperParamSearch, "character")

  checkIsClass(randomSample, c("numeric", "integer"))
  checkHigherEqual(randomSample, 1)

  checkIsClass(randomSampleSeed, c("numeric", "integer", "NULL"))

  est <- setEstimator(
    learningRate = 2e-3,
    weightDecay = 0.02,
    batchSize = 256L,
    epochs = 256L,
    device = device,
    optimizer = torch$optim$AdamW,
    # keep scheduler object for other models; we'll bypass it for RealMLP internally
    scheduler = list(
      fun = torch$optim$lr_scheduler$ReduceLROnPlateau,
      params = list(patience = 1000000L)
    ),
    criterion = torch$nn$BCEWithLogitsLoss, # logits + label smoothing inside Estimator
    earlyStopping = NULL, # train full and pick best
    compile = FALSE,
    metric = "loss",
    seed = NULL,
    trainValidationSplit = FALSE
  )
  # RealMLP-specific knobs passed through estimatorSettings
  est$beta2 <- 0.95
  est$eps <- 1e-8
  est$labelSmoothing <- labelSmoothing
  est$lrSchedule <- "coslog4"
  est$dropoutSchedule <- "flat_cos"
  est$weightDecaySchedule <- "flat_cos"
  est$scalingLrMult <- scalingLrMult
  est$biasLrMult <- biasLrMult
  est$actLrMult <- actLrMult
  est$embeddingLrMult <- embeddingLrMult
  est$biasWdFactor <- 0.0
  est$dataDependentInit <- TRUE
  est$dataDependentInitBatches <- 8L
  est$dataDependentInitMode <- dataDependentInitMode
  est$dataDependentInitTargetVar <- dataDependentInitTargetVar
  est$dataDependentInitMaxRows <- as.integer(dataDependentInitMaxRows)
  est$dataDependentInitGainClip <- dataDependentInitGainClip
  est$dataDependentInitBiasMode <- "he5"
  est$dataDependentInitBiasScale <- 1.0
  est$dataDependentInitBiasRefitSteps <- as.integer(dataDependentInitBiasRefitSteps)
  est$paramsToTune <- extractParamsToTune(est)

  paramGrid <- list(
    numLayers = as.integer(numLayers),
    sizeHidden = as.integer(sizeHidden),
    dropout = dropout,
    sizeEmbedding = as.integer(sizeEmbedding),
    numericEmbeddingMode = numericEmbeddingMode,
    numericNumFrequencies = as.integer(numericNumFrequencies),
    numericPeriodicInitStd = numericPeriodicInitStd,
    numericPbldHiddenDim = as.integer(numericPbldHiddenDim),
    numericPbldEmbeddingDim = as.integer(numericPbldEmbeddingDim),
    paperMode = paperMode,
    tokenAggregation = tokenAggregation,
    featureScaleMode = featureScaleMode
  )
  paramGrid <- c(paramGrid, est$paramsToTune)

  postProcess <- function(x) {
    if (x$tokenAggregation == "auto") {
      x$tokenAggregation <- if (isTRUE(x$paperMode)) "sum" else "mean"
    }
    if (x$featureScaleMode == "auto") {
      x$featureScaleMode <- "scalar"
    }
    x
  }

  results <- createDeepModelSettings(
    paramDefinition = paramGrid,
    estimatorSettings = est,
    modelParamNames = c(
      "numLayers", "sizeHidden", "dropout", "sizeEmbedding",
      "numericEmbeddingMode", "numericNumFrequencies",
      "numericPeriodicInitStd", "numericPbldHiddenDim",
      "numericPbldEmbeddingDim", "paperMode", "tokenAggregation",
      "featureScaleMode"
    ),
    modelType = "RealMLP",
    hyperParamSearch = hyperParamSearch,
    randomSample = randomSample,
    randomSampleSeed = randomSampleSeed,
    postProcess = postProcess,
    legacySearchExplicit = legacySearchExplicit
  )
  results
}
