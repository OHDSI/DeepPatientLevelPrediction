#' lrPerBatch <- torch::lr_scheduler(
#'   "lrPerBatch",
#'   initialize = function(
#'     optimizer,
#'     startLR = 1e-7,
#'     endLR = 1.0,
#'     nIters = 100,
#'     lastEpoch = -1,
#'     verbose = FALSE) {
#' 
#'     self$optimizer <- optimizer
#'     self$endLR <- endLR
#'     self$base_lrs <- startLR
#'     self$iterations <- nIters
#'     self$last_epoch <- lastEpoch
#'     self$multiplier <- (endLR/startLR)^(1/nIters)
#' 
#'     super$initialize(optimizer, last_epoch=lastEpoch, verbose)
#' 
#'   },
#' 
#'   get_lr = function() {
#'     if (self$last_epoch > 0) {
#'       lrs <- numeric(length(self$optimizer$param_groups))
#'       for (i in seq_along(self$optimizer$param_groups)) {
#'         lrs[i] <- self$base_lrs[[i]] * (self$endLR / self$base_lrs[[i]]) ^ (self$last_epoch/(self$iterations-1))
#'       }
#'     } else {
#'       lrs <- as.numeric(self$base_lrs)
#'     }
#'     lrs
#'   }
#' 
#' )
#' 
#' #' Find learning rate that decreases loss the most
#' #' @description Method originated from https://arxiv.org/abs/1506.01186 but this
#' #' implementation draws inspiration from various other implementations such as
#' #' pytorch lightning, fastai, luz and pytorch-lr-finder.
#' #' @param dataset torch dataset, training dataset
#' #' @param modelType the function used to initialize the model
#' #' @param modelParams parameters used to initialize model
#' #' @param estimatorSettings settings for estimator to fit model
#' #' @param minLR lower bound of learning rates to search through
#' #' @param maxLR upper bound of learning rates to search through
#' #' @param numLR number of learning rates to go through
#' #' @param smooth smoothing to use on losses
#' #' @param divergenceThreshold if loss increases this amount above the minimum, stop.
#' #' @export
#' lrFinder <- function(dataset, modelType, modelParams, estimatorSettings,
#'                      minLR=1e-7, maxLR=1, numLR=100, smooth=0.05,
#'                      divergenceThreshold=4) {
#'     torch::torch_manual_seed(seed=estimatorSettings$seed)
#'     model <- do.call(modelType, modelParams)
#'     if (is.function(estimatorSettings$device)) {
#'       device = estimatorSettings$device()
#'     } else {device = estimatorSettings$device}
#'     model$to(device=device)
#' 
#'     optimizer <- estimatorSettings$optimizer(model$parameters, lr=minLR)
#' 
#'     # made a special lr scheduler for this task
#'     scheduler <- lrPerBatch(optimizer = optimizer,
#'                             startLR = minLR,
#'                             endLR = maxLR,
#'                             nIters = numLR)
#' 
#'     criterion <- estimatorSettings$criterion()
#' 
#'     batchIndex <- seq(length(dataset))
#'     set.seed(estimatorSettings$seed)
#' 
#'     losses <- numeric(numLR)
#'     lrs <- numeric(numLR)
#'     ParallelLogger::logInfo('\nSearching for best learning rate')
#'     progressBar <- utils::txtProgressBar(style = 3)
#'     for (i in seq(numLR)) {
#'       optimizer$zero_grad()
#' 
#'       batch <- dataset[sample(batchIndex, estimatorSettings$batchSize)]
#'       batch <- batchToDevice(batch, device=device)
#' 
#'       output <- model(batch$batch)
#' 
#'       loss <- criterion(output, batch$target)
#'       if (!is.null(smooth) && i != 1) {
#'         losses[i] <- smooth * loss$item() + (1 - smooth) * losses[i-1]
#'       } else {
#'         losses[i] <- loss$item()
#'       }
#'       lrs[i] <- optimizer$param_groups[[1]]$lr
#' 
#'       loss$backward()
#'       optimizer$step()
#'       scheduler$step()
#'       utils::setTxtProgressBar(progressBar, i / numLR)
#' 
#'       if (i == 1) {
#'         bestLoss <- losses[i]
#'       } else {
#'         if (losses[i] < bestLoss) {
#'           bestLoss <- losses[i]
#'         }
#'       }
#' 
#'       if (losses[i] > (divergenceThreshold * bestLoss)) {
#'         ParallelLogger::logInfo("\nLoss diverged - stopped early")
#'         break
#'       }
#' 
#'     }
#' 
#'     # find LR where gradient is highest but before global minimum is reached
#'     # I added -5 to make sure it is not still in the minimum
#'     globalMinimum <- which.min(losses)
#'     grad <- as.numeric(torch::torch_diff(torch::torch_tensor(losses[1:(globalMinimum-5)])))
#'     smallestGrad <- which.min(grad)
#' 
#'     suggestedLR <- lrs[smallestGrad]
#' 
#'   return(suggestedLR)
#' }