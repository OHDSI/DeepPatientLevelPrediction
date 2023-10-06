#' TrainingCache
#' @description
#' Parameter caching for training persistence and continuity
#' @export
TrainingCache <- R6::R6Class(
  "TrainingCache",
  
  private = list(
    .paramPersistence = list(
      gridSearchPredictions = NULL,
      modelParams = NULL,
      gridPerformance = NULL
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
          return(which(sapply(private$.paramPersistence$gridSearchPredictions,
            is.null))[1])
        }
      }
    },
    
    #' @description
    #' Remove the training cache from the analysis path
    dropCache = function() {
      # TODO
    }
  )
)
