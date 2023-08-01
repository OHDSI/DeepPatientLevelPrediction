catTensor <- torch::torch_tensor(cbind(dataCat$rowId, 
                                       dataCat$timeId + 1, 
                                       dataCat$columnId),
                                 dtype=torch::torch_long())
chunks <- as.integer(torch::torch_unique_consecutive(catTensor[,1], 
                                                     return_counts = TRUE)[[3]])
start <- Sys.time()
times <- torch::torch_split(catTensor[,2], chunks)
delta <- Sys.time() - start


times <- lapply(times, function(x) torch::torch_unique_consecutive(x)[[1]])
self$temporalData$times <- as.list(integer(length(self$target)))
self$temporalData$times[unique(dataCat$rowId)] <- times

self$temporalData$lengths <- unlist(lapply(self$temporalData$times, function(x) {if (!is.integer(x)) {x$shape[[1]]}
  else 0L}))
self$temporalData$lengths <- torch::torch_tensor(self$temporalData$lengths, dtype=torch::torch_float32())

# maxVisit, if Patient has more visits than this the oldest are truncated
# to save computations
self$maxVisit <- torch::torch_quantile(self$temporalData$lengths, 0.99)$item()
self$temporalData$lengths <- torch::torch_tensor(self$temporalData$lengths, dtype=torch::torch_long())
self$temporalData$lengths[self$temporalData$lengths>self$maxVisit] <- self$maxVisit
browser()