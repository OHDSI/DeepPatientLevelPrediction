#' Estimator
#' @description 
#' A generic R6 class that wraps around a torch nn module and can be used to 
#' fit and predict the model defined in that module.
#' @export
Estimator <- R6::R6Class(
  classname = 'Estimator',
  lock_objects = FALSE,
  public = list(
    #' @description 
    #' Creates a new estimator
    #' @param baseModel       The torch nn module to use as model
    #' @param modelParameters Parameters to initialize the baseModel
    #' @param fitParameters   Parameters required for the estimator fitting
    #' @param optimizer       A torch optimizer to use, default is Adam
    #' @param criterion       The torch loss function to use, defaults to 
    #'                        binary cross entropy with logits
    #'@param device           Which device to use for fitting, default is cpu
    #'@param patience         Patience to use for early stopping                      
    initialize = function(baseModel, 
                          modelParameters, 
                          fitParameters,
                          optimizer=torch::optim_adam,
                          criterion=torch::nn_bce_with_logits_loss,
                          scheduler=torch::lr_reduce_on_plateau,
                          device='cpu') {
      self$device <- device
      self$model <- do.call(baseModel, modelParameters)
      self$modelParameters <- modelParameters
      self$fitParameters <- fitParameters
      self$epochs <- self$itemOrDefaults(fitParameters, 'epochs', 10)
      self$learningRate <- self$itemOrDefaults(fitParameters,'learningRate', 1e-3)
      self$l2Norm <- self$itemOrDefaults(fitParameters, 'weightDecay', 1e-5)
      self$batchSize <- self$itemOrDefaults(fitParameters, 'batchSize', 1024)
      self$posWeight <- self$itemOrDefaults(fitParameters, 'posWeight', 1)
      
      self$previousEpochs <- self$itemOrDefaults(fitParameters, 'previousEpochs', 0)
      self$model$to(device=self$device)
      
      self$optimizer <- optimizer(params=self$model$parameters, 
                                  lr=self$learningRate, 
                                  weight_decay=self$l2Norm)
      self$criterion <- criterion()
      
      self$scheduler <- scheduler(self$optimizer, patience=1,
                                  verbose=TRUE)
      
    },
    
    #' @description fits the estimator
    #' @param dataset     a torch dataset to use for model fitting
    fit = function(dataset) {
      valLosses <- c()
      valAUCs <- c()
      
      batchIndex <- torch::torch_randperm(length(dataset)) + 1L
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex)/self$batchSize))
      
      times <- list()
      for (epochI in 1:self$epochs) {
        # fit the model
        startTime <- Sys.time()
        trainLoss <- self$fitEpoch(dataset, batchIndex)
        endTime <- Sys.time()
        
        delta <- endTime - startTime
        currentEpoch <- epochI + self$previousEpochs
        ParallelLogger::logInfo('Epochs: ', currentEpoch, 
                                ' | Train Loss: ', round(trainLoss,3),
                                ' | Time: ', round(delta, 3), ' ', 
                                units(delta))
        times <- c(times, round(delta, 3))
        torch::cuda_empty_cache()
              }
      ParallelLogger::logInfo('Average time per epoch was: ', round(mean(as.numeric(times)),3), ' ' , units(delta))
      invisible(self)
    },
    
    #' @description 
    #' fits estimator for one epoch (one round through the data)
    #' @param dataset     torch dataset to use for fitting
    #' @param batchIndex  indices of batches 
    fitEpoch = function(dataset, batchIndex){
      trainLosses <- torch::torch_empty(length(batchIndex))
      ix <- 1
      self$model$train()
      progressBar <- utils::txtProgressBar(style=3)
      coro::loop(for (b in batchIndex) {
        self$optimizer$zero_grad()
        batch <- self$batchToDevice(dataset[b])
        out <- self$model(batch[[1]])
        loss <- self$criterion(out, batch[[2]])
        loss$backward()
        
        self$optimizer$step()
        trainLosses[ix] <- loss$detach()
        
        utils::setTxtProgressBar(progressBar, ix/length(batchIndex))
        ix <- ix + 1
      })
      close(progressBar)
      trainLosses$mean()$item()
    },
    
    #' @description 
    #' sends a batch of data to device
    #' assumes batch includes lists of tensors to arbitrary nested depths
    #' @param batch the batch to send, usually a list of torch tensors
    #' @return the batch on the required device
    batchToDevice = function(batch) {
      if (class(batch)[1] == 'torch_tensor') {
        batch <- batch$to(device=self$device)
      } else {
        ix <- 1
        for (b in batch) {
          if (class(b)[1] == 'torch_tensor') {
            b <- b$to(device=self$device)
          } else {
            b <- self$batchToDevice(b)
          }
          if (!is.null(b)) {
            batch[[ix]] <- b
          }
          ix <- ix + 1
        }
      }
      return(batch)
    },
    
    #' @description
    #' select item from list, and if it's null sets a default
    #' @param list A list with items
    #' @param item Which list item to retrieve
    #' @param default The value to return if list doesn't have item
    #' @return the list item or default 
    itemOrDefaults = function (list, item, default = NULL) {
      value <- list[[item]]
      if (is.null(value)) default else value
    }
  )
)