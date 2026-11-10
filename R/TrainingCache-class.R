#' TrainingCache
#'
#' Stores hyperparameter-search progress so interrupted model training can be
#' resumed from an analysis directory.
#'
#' @return An R6 class generator for persistent training caches.
#'
#' @examples
#' cacheDirectory <- tempfile("training-cache-")
#' dir.create(cacheDirectory)
#' cache <- trainingCache$new(cacheDirectory)
#' cache$saveModelParams(list(list(sizeHidden = 64)))
#' cache$isParamGridIdentical(list(list(sizeHidden = 64)))
#' unlink(cacheDirectory, recursive = TRUE)
#'
#' @importFrom R6 R6Class
#' @export
trainingCache <- R6Class(
  "TrainingCache",
  private = list(
    .paramPersistence = list(
      gridSearchPredictions = NULL,
      modelParams = NULL,
      fingerprint = NULL
    ),
    .paramContinuity = list(),
    .saveDir = NULL,
    writeToFile = function() {
      saveRDS(private$.paramPersistence, file.path(private$.saveDir))
    },
    readFromFile = function() {
      private$.paramPersistence <- readRDS(file.path(private$.saveDir))
    }
  ),
  public = list(
    #' @description
    #' Creates a new training cache
    #' @param inDir Path to the analysis directory
    initialize = function(inDir) {
      private$.saveDir <- file.path(inDir, "paramPersistence.rds")

      if (file.exists(private$.saveDir)) {
        private$readFromFile()
      } else {
        private$writeToFile()
      }
    },

    #' @description
    #' Checks whether the parameter grid in the model settings is identical to
    #' the cached parameters.
    #' @param inModelParams Parameter grid from the model settings
    #' @returns Whether the provided and cached parameter grid is identical
    isParamGridIdentical = function(inModelParams) {
      return(identical(inModelParams, private$.paramPersistence$modelParams))
    },

    #' @description
    #' Checks whether the cached results were produced with the same data and
    #' training configuration.
    #' @param inFingerprint Fingerprint of the current data and configuration
    #' @returns Whether the provided and cached fingerprint are identical
    isFingerprintIdentical = function(inFingerprint) {
      return(identical(inFingerprint, private$.paramPersistence$fingerprint))
    },

    #' @description
    #' Saves the fingerprint of the data and configuration to the training cache
    #' @param inFingerprint Fingerprint of the current data and configuration
    saveFingerprint = function(inFingerprint) {
      private$.paramPersistence$fingerprint <- inFingerprint
      private$writeToFile()
    },

    #' @description
    #' Saves the grid search results to the training cache
    #' @param inGridSearchPredictions Grid search predictions
    saveGridSearchPredictions = function(inGridSearchPredictions) {
      private$.paramPersistence$gridSearchPredictions <-
        inGridSearchPredictions
      private$writeToFile()
    },

    #' @description
    #' Saves the parameter grid to the training cache
    #' @param inModelParams Parameter grid from the model settings
    saveModelParams = function(inModelParams) {
      private$.paramPersistence$modelParams <- inModelParams
      private$writeToFile()
    },

    #' @description
    #' Gets the grid search results from the training cache
    #' @returns Grid search results from the training cache
    getGridSearchPredictions = function() {
      return(private$.paramPersistence$gridSearchPredictions)
    },

    #' @description
    #' Check if cache is full
    #' @returns A logical value.
    isFull = function() {
      return(all(unlist(lapply(
        private$.paramPersistence$gridSearchPredictions,
        function(x) !is.null(x$gridPerformance)
      ))))
    },

    #' @description
    #' Gets the last index from the cached grid search
    #' @returns Last grid search index
    getLastGridSearchIndex = function() {
      if (is.null(private$.paramPersistence$gridSearchPredictions)) {
        return(1)
      } else {
        # if only a single hyperparameter combination is assessed return 1
        if (length(private$.paramPersistence$gridSearchPredictions) == 1) {
          return(1)
        } else {
          return(which(sapply(
            private$.paramPersistence$gridSearchPredictions,
            is.null
          ))[1])
        }
      }
    },

    #' @description
    #' Remove the training cache from the analysis path
    dropCache = function() {
      # TODO
    },
    
    #' @description
    #' Trims the performance of the hyperparameter results by removing 
    #' the predictions from all but the best performing hyperparameter
    #' @param hyperparameterResults List of hyperparameter results 
    #' @param maximize Whether higher performance values are better
    trimPerformance = function(hyperparameterResults, maximize = TRUE) {
      indexOfBest <- getBestGridIndex(hyperparameterResults, maximize)
      if (length(indexOfBest) != 0) {
        for (i in seq_along(hyperparameterResults)) {
          if (!is.null(hyperparameterResults[[i]]) && i != indexOfBest) {
            hyperparameterResults[[i]]$prediction <- list(NULL)
          }
        }
        ParallelLogger::logInfo(
          paste0(
            "Caching all grid search results and
                                     prediction for best combination ",
            indexOfBest
          )
        )
      }
      return(hyperparameterResults)
    }
  )
)

setupCache <- function(analysisPath, parameters, fingerprint = NULL) {
  trainCache <- trainingCache$new(analysisPath)
  isSameRun <- trainCache$isParamGridIdentical(parameters) &&
    trainCache$isFingerprintIdentical(fingerprint)
  if (!isSameRun) {
    cached <- trainCache$getGridSearchPredictions()
    if (any(!vapply(cached, is.null, logical(1)))) {
      ParallelLogger::logInfo(
        "Ignoring the training cache because the data or settings have changed"
      )
    }
    hyperparameterResults <- list()
    length(hyperparameterResults) <- length(parameters)
    trainCache$saveGridSearchPredictions(hyperparameterResults)
    trainCache$saveModelParams(parameters)
    trainCache$saveFingerprint(fingerprint)
  }
  return(trainCache)
}

# Identifies everything besides the parameter grid that cached results depend on.
createCacheFingerprint <- function(
    dataset,
    labels,
    modelSettings,
    tuningMetric = NULL) {
  estimatorSettings <- modelSettings$estimatorSettings
  # a seed drawn because none was requested should not block resuming a run
  if (identical(estimatorSettings$seed, attr(estimatorSettings, "generatedSeed"))) {
    estimatorSettings$seed <- NULL
  }
  # the device changes where the model trains, not what is being estimated
  estimatorSettings$device <- NULL
  labelColumns <- intersect(c("rowId", "outcomeCount", "index"), names(labels))
  return(rlang::hash(list(
    data = as.character(dataset$fingerprint()),
    labels = normalizeForFingerprint(as.list(as.data.frame(labels)[labelColumns])),
    modelType = modelSettings$modelType,
    estimatorSettings = normalizeForFingerprint(estimatorSettings),
    tuningMetric = describeTuningMetric(tuningMetric),
    files = describeInputFiles(estimatorSettings)
  )))
}

# Files read during training, identified by their contents, since the same
# path can hold new weights or embeddings in a later run.
describeInputFiles <- function(estimatorSettings) {
  paths <- estimatorSettings$finetuneModelPath
  strategy <- estimatorSettings$initStrategy
  if (inherits(strategy, "python.builtin.object") &&
    reticulate::py_has_attr(strategy, "embedding_file")) {
    paths <- c(paths, as.character(strategy$embedding_file))
  }
  return(vapply(
    paths,
    function(path) if (file.exists(path)) rlang::hash_file(path) else NA_character_,
    character(1),
    USE.NAMES = FALSE
  ))
}

# createTuningMetric() wraps the metric function in a closure that holds the
# original function and its funArgs, so the wrapper alone tells metrics apart
# only by name.
describeTuningMetric <- function(tuningMetric) {
  if (is.null(tuningMetric)) {
    return(NULL)
  }
  metricFunction <- tuningMetric$fun
  if (is.function(metricFunction) &&
    all(c("fun", "funArgs") %in% ls(environment(metricFunction)))) {
    metricFunction <- environment(metricFunction)$fun
  }
  return(normalizeForFingerprint(list(
    name = tuningMetric$name,
    maximize = tuningMetric$maximize,
    fun = metricFunction,
    funArgs = tuningMetric$funArgs
  )))
}

# Python objects and closures have no stable serialized form. Values that R
# closures or Python callables capture are not part of the description, so
# changing such a value alone does not invalidate the cache.
normalizeForFingerprint <- function(x) {
  if (inherits(x, "python.builtin.object")) {
    return(describePythonObject(x))
  }
  if (inherits(x, "delayed")) {
    return(normalizeForFingerprint(x()))
  }
  if (is.function(x)) {
    return(deparse(x, control = NULL))
  }
  if (is.list(x)) {
    return(lapply(x, normalizeForFingerprint))
  }
  if (is.atomic(x) && !is.null(x)) {
    # equal vectors can hash differently when one is a compact ALTREP sequence
    # such as 1:n; c() gives an ordinary copy
    attributes <- attributes(x)
    x <- c(x)
    attributes(x) <- attributes
  }
  return(x)
}

describePythonObject <- function(x) {
  builtins <- reticulate::import_builtins(convert = FALSE)
  description <- as.character(builtins$repr(x))
  if (!reticulate::import("inspect")$isclass(x) &&
    reticulate::py_has_attr(x, "__dict__")) {
    description <- c(description, as.character(builtins$repr(builtins$vars(x))))
  }
  return(gsub(" at 0x[0-9a-fA-F]+", "", description))
}
