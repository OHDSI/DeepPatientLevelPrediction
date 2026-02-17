# @file: RealMLP.R

#' setRealMLP
#' Create settings for the RealMLP model (binary classification)
#' Preprocessing (robust scaling + clipping) is assumed to be handled upstream.
#'
#' @param numLayers  hidden layers (default 3)
#' @param sizeHidden hidden width (default 256)
#' @param dropout    base dropout p (default 0.15; scheduled with flat_cos)
#' @param sizeEmbedding embedding dim for compatibility with existing Embedding (default 64)
#' @param labelSmoothing epsilon for BCE label smoothing (default 0.1)
#' @param scalingLrMult LR multiplier for scaling parameters (default 6.0)
#' @param biasLrMult LR multiplier for bias parameters (default 0.1)
#' @param actLrMult LR multiplier for parametric activation parameters (default 0.1)
#' @param device     "cpu" or "cuda" (default "cpu")
#' @export
setRealMLP <- function(
  numLayers = 3L,
  sizeHidden = 256L,
  dropout = 0.15,
  sizeEmbedding = 64L,
  labelSmoothing = 0.1,
  scalingLrMult = 6.0,
  biasLrMult = 0.1,
  actLrMult = 0.1,
  device = "cpu"
) {
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
  if (labelSmoothing > 1) {
    stop("labelSmoothing needs to be <= 1")
  }
  checkIsClass(scalingLrMult, "numeric")
  checkHigherEqual(scalingLrMult, 0)
  checkIsClass(biasLrMult, "numeric")
  checkHigherEqual(biasLrMult, 0)
  checkIsClass(actLrMult, "numeric")
  checkHigherEqual(actLrMult, 0)
  checkIsClass(device, c("character", "function"))

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
    metric = "auc",
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
  est$biasWdFactor <- 0.0

  param <- list(
    numLayers = as.integer(numLayers),
    sizeHidden = as.integer(sizeHidden),
    dropout = dropout,
    sizeEmbedding = as.integer(sizeEmbedding)
  )

  results <- list(
    fitFunction = "DeepPatientLevelPrediction::fitEstimator",
    param = list(param), # fixed tuned defaults; no grid/HPO
    estimatorSettings = est,
    saveType = "file",
    modelParamNames = c("numLayers", "sizeHidden", "dropout", "sizeEmbedding"),
    modelType = "RealMLP"
  )
  attr(results$param, "settings")$modelType <- results$modelType
  class(results) <- "modelSettings"
  results
}
