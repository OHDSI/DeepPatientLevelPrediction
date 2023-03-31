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
    initialize = function(inDir) {
      private$.saveDir <- file.path(inDir, "paramPersistence.rds")
      
      if (file.exists(private$.saveDir)) {
        private$readFromFile()
      } else {
        private$writeToFile()
      }
    },
    
    isPersistent = function(searchParam) {
      return(identical(searchParam, private$.paramPersistence$modelParams))
    },
    
    saveGridSearchPredictions = function(inGridSearchPredictions) {
      private$.paramPersistence$gridSearchPredictions <-
        inGridSearchPredictions
      private$writeToFile()
    },
    
    saveModelParams = function(modelParams) {
      private$.paramPersistence$modelParams <- modelParams
      private$writeToFile()
    },
    
    getGridSearchPredictions = function() {
      return(private$.paramPersistence$gridSearchPredictions)
    },
    
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
    }
  )
)
