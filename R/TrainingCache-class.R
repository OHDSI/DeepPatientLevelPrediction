#' TrainingCache
#' @description
#' Parameter caching for training persistence and continuity
#' @export
trainingCache <- R6::R6Class(
  "TrainingCache",
  private = list(
    .paramPersistence = list(
      gridSearchPredictions = NULL,
      modelParams = NULL,
      cacheVersion = 1L,
      searchResults = NULL,
      searchModelParams = NULL,
      candidatePool = NULL,
      searchHistory = NULL,
      generatorState = NULL,
      searchComplete = FALSE
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
    #' @returns Boolen
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
    #' @param maximize Whether higher metric values are better
    trimPerformance = function(hyperparameterResults, maximize = TRUE) {
      values <- unlist(
        lapply(hyperparameterResults,
               function(x)
                 x$gridPerformance$cvPerformance)
      )
      indexOfBest <- if (isTRUE(maximize)) {
        which.max(values)
      } else {
        which.min(values)
      }
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
    },

    #' @description
    #' Checks whether a PLP-style hyperparameter search matches the cache.
    #' @param inModelParams Search identity object
    #' @returns Whether the cached search identity is identical
    isSearchIdentical = function(inModelParams) {
      return(identical(
        inModelParams,
        private$.paramPersistence$searchModelParams
      ))
    },

    #' @description
    #' Saves the PLP-style hyperparameter search identity.
    #' @param inModelParams Search identity object
    saveSearchModelParams = function(inModelParams) {
      private$.paramPersistence$cacheVersion <- 2L
      private$.paramPersistence$searchModelParams <- inModelParams
      private$writeToFile()
    },

    #' @description
    #' Saves candidate pool for PLP-style tuning.
    #' @param candidatePool Candidate pool
    saveCandidatePool = function(candidatePool) {
      private$.paramPersistence$cacheVersion <- 2L
      private$.paramPersistence$candidatePool <- candidatePool
      private$writeToFile()
    },

    #' @description
    #' Gets candidate pool for PLP-style tuning.
    #' @returns Candidate pool
    getCandidatePool = function() {
      private$.paramPersistence$candidatePool
    },

    #' @description
    #' Saves PLP-style search results.
    #' @param inSearchResults Search results
    saveSearchResults = function(inSearchResults) {
      private$.paramPersistence$cacheVersion <- 2L
      private$.paramPersistence$searchResults <- inSearchResults
      private$.paramPersistence$gridSearchPredictions <- inSearchResults
      private$writeToFile()
    },

    #' @description
    #' Gets PLP-style search results.
    #' @returns Search results
    getSearchResults = function() {
      private$.paramPersistence$gridSearchPredictions %||%
        private$.paramPersistence$searchResults
    },

    #' @description
    #' Checks whether PLP-style search is complete.
    #' @returns Boolean
    isSearchFull = function() {
      results <- self$getSearchResults()
      if (is.null(results) || length(results) == 0) {
        return(FALSE)
      }
      all(unlist(lapply(
        results,
        function(x) !is.null(x) && !is.null(x$gridPerformance)
      )))
    },

    #' @description
    #' Gets next candidate index for PLP-style search.
    #' @returns Candidate index
    getNextSearchIndex = function() {
      results <- self$getSearchResults()
      if (is.null(results) || length(results) == 0) {
        return(1L)
      }
      nextIndex <- which(sapply(results, is.null))[1]
      if (is.na(nextIndex)) {
        return(length(results) + 1L)
      }
      nextIndex
    },

    #' @description
    #' Saves PLP-style adaptive search state.
    #' @param inSearchResults Search results
    #' @param inHistory Tuning history
    #' @param inGeneratorState Serialized generator state
    #' @param complete Whether the search is complete
    saveAdaptiveSearchState = function(
        inSearchResults,
        inHistory,
        inGeneratorState,
        complete = FALSE) {
      private$.paramPersistence$cacheVersion <- 2L
      private$.paramPersistence$searchResults <- inSearchResults
      private$.paramPersistence$gridSearchPredictions <- inSearchResults
      private$.paramPersistence$searchHistory <- inHistory
      private$.paramPersistence$generatorState <- inGeneratorState
      private$.paramPersistence$searchComplete <- isTRUE(complete)
      private$writeToFile()
    },

    #' @description
    #' Gets adaptive search history.
    #' @returns Tuning history
    getSearchHistory = function() {
      private$.paramPersistence$searchHistory %||% list()
    },

    #' @description
    #' Gets cached custom generator state.
    #' @returns Generator state
    getGeneratorState = function() {
      private$.paramPersistence$generatorState
    },

    #' @description
    #' Checks whether adaptive search is complete.
    #' @returns Boolean
    isAdaptiveSearchComplete = function() {
      isTRUE(private$.paramPersistence$searchComplete)
    }
  )
)

sanitizeHyperparameterSettings <- function(hyperparameterSettings) {
  generatorIdentity <- NULL
  if (!is.null(hyperparameterSettings$generator)) {
    generatorIdentity <- if (is.function(hyperparameterSettings$generator)) {
      hyperparameterSettings$generator
    } else {
      class(hyperparameterSettings$generator)
    }
  }
  list(
    search = hyperparameterSettings$search,
    sampleSize = hyperparameterSettings$sampleSize,
    randomSeed = hyperparameterSettings$randomSeed,
    tuningMetricName = hyperparameterSettings$tuningMetric$name,
    tuningMetricMaximize = hyperparameterSettings$tuningMetric$maximize,
    generator = generatorIdentity
  )
}

setupCache <- function(analysisPath, parameters) {
  trainCache <- trainingCache$new(analysisPath)
  if (trainCache$isParamGridIdentical(parameters)) {
    hyperparameterResults <- trainCache$getGridSearchPredictions()
  } else {
    hyperparameterResults <- list()
    length(hyperparameterResults) <- length(parameters)
    trainCache$saveGridSearchPredictions(hyperparameterResults)
    trainCache$saveModelParams(parameters)
  }
  return(trainCache)
}

setupSearchCache <- function(
    analysisPath,
    paramDefinition,
    hyperparameterSettings,
    candidatePool) {
  trainCache <- trainingCache$new(analysisPath)
  searchIdentity <- list(
    paramDefinition = paramDefinition,
    hyperparameterSettings = sanitizeHyperparameterSettings(
      hyperparameterSettings
    )
  )
  if (trainCache$isSearchIdentical(searchIdentity)) {
    cachedPool <- trainCache$getCandidatePool()
    if (!is.null(cachedPool)) {
      candidatePool <- cachedPool
    }
  } else {
    hyperparameterResults <- list()
    length(hyperparameterResults) <- length(candidatePool)
    trainCache$saveSearchResults(hyperparameterResults)
    trainCache$saveSearchModelParams(searchIdentity)
    trainCache$saveCandidatePool(candidatePool)
  }
  return(trainCache)
}

setupAdaptiveSearchCache <- function(
    analysisPath,
    paramDefinition,
    hyperparameterSettings) {
  trainCache <- trainingCache$new(analysisPath)
  searchIdentity <- list(
    paramDefinition = paramDefinition,
    hyperparameterSettings = sanitizeHyperparameterSettings(
      hyperparameterSettings
    )
  )
  if (!trainCache$isSearchIdentical(searchIdentity)) {
    trainCache$saveAdaptiveSearchState(
      inSearchResults = list(),
      inHistory = list(),
      inGeneratorState = NULL,
      complete = FALSE
    )
    trainCache$saveSearchModelParams(searchIdentity)
  }
  trainCache
}
