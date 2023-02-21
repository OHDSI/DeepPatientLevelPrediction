lrPerBatch <- torch::lr_scheduler(
  "lrPerBatch",
  initialize = function(
    optimizer,
    startLR = 1e-7,
    endLR = 1.0,
    nIters = 100,
    lastEpoch = -1,
    verbose = FALSE) {
    
    self$optimizer <- optimizer
    self$endLR <- endLR
    self$base_lrs <- startLR
    self$iterations <- nIters
    self$last_epoch <- lastEpoch
    self$multiplier <- (endLR/startLR)^(1/nIters)
    
    super$initialize(optimizer, last_epoch=lastEpoch, verbose)
    
  },
  
  get_lr = function() {
    if (self$last_epoch > 0) {
      lrs <- numeric(length(self$optimizer$param_groups))
      for (i in seq_along(self$optimizer$param_groups)) {
        lrs[i] <- self$base_lrs[[i]] * (self$endLR / self$base_lrs[[i]]) ^ (self$last_epoch/(self$iterations-1))
      }
    } else {
      lrs <- as.numeric(self$base_lrs)
    }
    lrs
  }
  
)

lrFinder <- function(dataset, modelType, modelParams, fitParams,
                     minLR=1e-7, maxLR=1, numLR=100, smooth=0.05,
                     divergenceThreshold=4) {
    
    model <- do.call(modelType, modelParams)
    model$to(device='cuda:0')
    
    optimizer <- torch::optim_adam(model$parameters, lr=minLR)
    
    scheduler <- lrPerBatch(optimizer = optimizer,
                            startLR = minLR,
                            endLR = maxLR,
                            nIters = numLR)
    
    criterion <- torch::nn_bce_with_logits_loss()
    
    batchIndex <- seq(length(dataset))
    
    
    losses <- numeric(numLR)
    lrs <- numeric(numLR)
    ParallelLogger::logInfo('Searching for best learning rate')
    progressBar <- utils::txtProgressBar(style = 3)
    for (i in seq(numLR)) {
      optimizer$zero_grad()
      
      batch <- dataset[sample(batchIndex, fitParams$batchSize)]
      batch$batch$cat <- batch$batch$cat$to(device='cuda:0')
      batch$batch$num <- batch$batch$num$to(device='cuda:0')
      batch$target <- batch$target$to(device='cuda:0')
      
      output <- model(batch$batch)
      
      loss <- criterion(output, batch$target)
      if (!is.null(smooth) && i != 1) {
        losses[i] <- smooth * loss$item() + (1 - smooth) * losses[i-1]
      } else {
        losses[i] <- loss$item()  
      }
      lrs[i] <- optimizer$param_groups[[1]]$lr
      
      loss$backward()
      optimizer$step()
      scheduler$step()
      utils::setTxtProgressBar(progressBar, i / numLR)
      
      if (i == 1) {
        bestLoss <- losses[i]
      } else {
        if (losses[i] < bestLoss) {
          bestLoss <- losses[i]
        }
      }
      
      if (losses[i] > (divergenceThreshold * bestLoss)) {
        ParallelLogger::logInfo("Loss diverged - stopped early")
        break
      }
      
    }
    
    grad <- as.numeric(torch::torch_diff(torch::torch_tensor(losses[1:i])))
    smallestGrad <- which.min(grad)
    
    suggestedLR <- lrs[smallestGrad]
    
  return(suggestedLR) 
}