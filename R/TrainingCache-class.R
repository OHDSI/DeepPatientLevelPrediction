#' TrainingCache
#' @description
#' Parameter caching for training persistence and continuity
#' 
TrainingCache <- R6::R6Class(
  "TrainingCache",
  
  private = list(
    .paramPersistence = list(
      gridSearchPredictions = NULL,
      modelParams = NULL
    ),
    .paramContinuity = list(),
    .saveDir = NULL,
    
    writeToFile = function() {
      saveRDS(private$.paramPersistence, file.path(private$.saveDir))
    },
    
    readFromFile = function() {
      tryCatch({
        private$.paramPersistence <- readRDS(file.path(private$.saveDir))
      },
      error=function(cond) {
        message(paste("Cannot load ", file.path(private$.saveDir)))
      },
      finally = {
      })
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
    #' @param inDir Path to the analysis directory
    #' @returns Whether the provided and cached parameter grid is identical
    isParamGridIdentical = function(searchParam) {
      return(identical(searchParam, private$.paramPersistence$modelParams))
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
    #' @param modelParams Parameter grid
    saveModelParams = function(modelParams) {
      private$.paramPersistence$modelParams <- modelParams
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
        return(which(
          sapply(
            private$.paramPersistence$gridSearchPredictions,
            is.null
          )
        )[1])
      }
    },
    
    #' @description
    #' Remove the training cache from the analysis path
    dropCache = function() {
      # TODO
    }
  )
)
