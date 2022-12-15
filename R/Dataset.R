#'  A torch dataset
#' @export
Dataset <- torch::dataset(
  name = "myDataset",
  #' @param data           a dataframe like object with the covariates
  #' @param labels         a dataframe with the labels
  #' @param numericalIndex in what column numeric data is in (if any)
  initialize = function(data, labels = NULL, numericalIndex = NULL) {
    start <- Sys.time()
    # determine numeric
    if (is.null(numericalIndex)) {
      numericalIndex <- data %>%
        dplyr::group_by(columnId) %>%
        dplyr::summarise(n = dplyr::n_distinct(.data$covariateValue)) %>%
        dplyr::collect() %>% 
        dplyr::pull(n) > 1
      self$numericalIndex <- numericalIndex
    } else {
      self$numericalIndex <- numericalIndex
    }

    self$target <- self$processTargets(labels, data)
    
    self$processCategorical(data)
    
    self$processNumerical(data)
    
    delta <- Sys.time() - start
    ParallelLogger::logInfo("Data conversion for dataset took ", signif(delta, 3), " ", attr(delta, "units"))
    
  },
  
  processNumerical = function(data) {
    if (sum(self$numericalIndex) == 0) {
      self$num <- NULL
    } else {
      numericalData <- data %>%
        dplyr::filter(columnId %in% !!which(self$numericalIndex)) %>%
        dplyr::collect()
      numericalData <- numericalData %>%
        dplyr::group_by(columnId) %>%
        dplyr::mutate(newId = dplyr::cur_group_id())
      indices <- torch::torch_tensor(cbind(numericalData$rowId, numericalData$newId), dtype = torch::torch_long())$t_()
      values <- torch::torch_tensor(numericalData$covariateValue, dtype = torch::torch_float32())
      self$num <- torch::torch_sparse_coo_tensor(
        indices = indices,
        values = values,
        size = c(self$target$shape, sum(self$numericalIndex))
      )$to_dense()
    }
    
  },
  
  processCategorical = function(data) {
    # add features
    catColumns <- which(!self$numericalIndex)
    dataCat <- dplyr::filter(data, columnId %in% catColumns) %>%
      dplyr::arrange(columnId) %>%
      dplyr::group_by(columnId) %>%
      dplyr::collect() %>%
      dplyr::mutate(newColumnId = dplyr::cur_group_id()) %>%
      dplyr::ungroup() %>%
      dplyr::select(c("rowId", "newColumnId")) %>%
      dplyr::rename(columnId = newColumnId) %>% 
      dplyr::arrange(rowId)
    catTensor <- torch::torch_tensor(cbind(dataCat$rowId, dataCat$columnId))
    tensorList <- torch::torch_split(catTensor[,2], 
                                     as.integer(torch::torch_unique_consecutive(catTensor[,1], 
                                                                                return_counts = TRUE)[[3]]))
    totalList <- as.list(integer(length(self$target)))
    totalList[unique(dataCat$rowId)] <- tensorList
    self$lengths <- unlist(lapply(totalList, function(x) {if (!is.null(x)) {x$shape[[1]]} 
      else 0L}))
    self$cat <- torch::nn_utils_rnn_pad_sequence(tensorList, batch_first = T)
  },
  processTargets = function(labels, data) {
    # add labels if training (make 0 vector for prediction)
    if (!is.null(labels)) {
      target <- torch::torch_tensor(labels)
    } else {
      target <- torch::torch_tensor(rep(0, data %>% dplyr::distinct(rowId)
                                        %>% dplyr::collect() %>% nrow()))
    }
    return(target)
  },
  getNumericalIndex = function() {
    return(
      self$numericalIndex
    )
  },
  numCatFeatures = function() {
    return(
      sum(!self$numericalIndex)
    )
  },
  numNumFeatures = function() {
    if (!is.null(self$num)) {
      return(self$num$shape[2])
    } else {
      return(0)
    }
  },
  .getbatch = function(item) {
    if (length(item) == 1) {
      return(self$.getBatchSingle(item))
    } else {
      return(self$.getBatchRegular(item))
    }
  },
  .getBatchSingle = function(item) {
    # add leading singleton dimension since models expects 2d tensors
    batch <- list(
      cat = self$cat[item]$unsqueeze(1),
      num = self$num[item]$unsqueeze(1)
    )
    return(list(
      batch = batch,
      target = self$target[item]$unsqueeze(1)
    ))
  },
  .getBatchRegular = function(item) {
    batch <- list(
      cat = self$cat[item],
      num = self$num[item]
    )
    return(list(
      batch = batch,
      target = self$target[item]
    ))
  },
  .length = function() {
    self$target$size()[[1]] # shape[1]
  }
)


#'  A torch dataset for temporal data
#' @export
TemporalDataset <- torch::dataset(
  name = "TemporalDataset",
  #' @param data           a dataframe like object with the covariates
  #' @param labels         a dataframe with the labels
  #' @param numericalIndex in what column numeric data is in (if any)
  initialize = function(data, labels = NULL, numericalIndex = NULL) { 
    start <- Sys.time()
    # separate temporal and non-temporal
    nonTemporal <- data %>% dplyr::filter(is.na(timeId))
    temporalData <- data %>% dplyr::filter(!is.na(timeId))
    
    self$nonTemp <- self$processNonTemporal(nonTemporal)
    
    self$target <- self$processTarget(labels, data)
    
    self$processTemporal(temporalData)
    
    delta <- Sys.time() - start
    ParallelLogger::logInfo("Data conversion for dataset took ", signif(delta, 3), " ", attr(delta, "units"))
  },
  processTemporal = function(data) {
    numericalIndex <- data %>%
      dplyr::group_by(columnId) %>%
      dplyr::summarise(n = dplyr::n_distinct(.data$covariateValue)) %>%
      dplyr::collect() %>% 
      dplyr::pull(n) > 1
    
    # add features
    catColumns <- which(!numericalIndex)
    dataCat <- dplyr::filter(data, columnId %in% catColumns) %>%
      dplyr::arrange(columnId) %>%
      dplyr::group_by(columnId) %>%
      dplyr::collect() %>%
      dplyr::mutate(newColumnId = dplyr::cur_group_id()) %>%
      dplyr::ungroup() %>%
      dplyr::select(c("rowId", "newColumnId", "timeId")) %>%
      dplyr::rename(columnId = newColumnId) %>%
      dplyr::arrange(rowId, timeId, columnId)
    catTensor <- torch::torch_tensor(cbind(dataCat$rowId, dataCat$timeId, dataCat$columnId))
    tensorList <- torch::torch_split(catTensor[,3], 
                                     as.integer(torch::torch_unique_consecutive(catTensor[,1], 
                                                                                return_counts = TRUE)[[3]]))
    totalList <- as.list(integer(length(self$target)))
    totalList[unique(dataCat$rowId)] <- tensorList
    
    times <- torch::torch_split(catTensor[,2], 
                                as.integer(torch::torch_unique_consecutive(catTensor[,1], 
                                                                           return_counts = TRUE)[[3]]))
    self$temp$times <- as.list(integer(length(self$target)))
    self$temp$times[unique(dataCat$rowId)] <- times
    
    self$temp$lengths <- unlist(lapply(totalList, function(x) {if (!is.integer(x)) {x$shape[[1]]}
      else 0L}))
    self$temp$lengths <- torch::torch_tensor(self$lengths, dtype=torch::torch_long())
    self$temp$cat <- torch::nn_utils_rnn_pad_sequence(totalList, batch_first = T)
    if (sum(numericalIndex) == 0) {
      self$temp$num <- NULL
    } else {
      stop(paste0('Currently the package does not support temporal continous covariates'))
    }
  },
  processNonTemporal = function(data) {
    # separate nonTemporal into numeric and non-numeric
    numericalIndex <- data %>%
      dplyr::group_by(columnId) %>%
      dplyr::summarise(n = dplyr::n_distinct(.data$covariateValue)) %>%
      dplyr::collect() %>% 
      dplyr::pull(n) > 1
    index <- data %>% 
      dplyr::distinct(covariateId) %>% 
      dplyr::collect() %>% 
      dplyr::pull()
    numericalCovariateIds <- index[which(numericalIndex)]
    categoricalCovariateIds <- index[which(!numericalIndex)]
    
    cat <- data %>% dplyr::filter(covariateId %in% categoricalCovariateIds) 
    num <- data %>% dplyr::filter(covariateId %in% numericalCovariateIds)
    
    return(list(cat=cat, num=num))
  },
  
  processTarget = function(labels, data) {
    # add labels if training (make 0 vector for prediction)
    if (!is.null(labels)) {
      target <- torch::torch_tensor(labels)
    } else {
      target <- torch::torch_tensor(rep(0, data %>% dplyr::distinct(rowId)
                                         %>% dplyr::collect() %>% nrow()))
    }
    return(target)
  },
  
  getNumericalIndex = function() {
    return(
      self$numericalIndex
    )
  },
  numCatFeatures = function() {
    return(
      sum(!self$numericalIndex)
    )
  },
  numNumFeatures = function() {
    if (!is.null(self$num)) {
      return(self$num$shape[2])
    } else {
      return(0)
    }
  },
  .getbatch = function(item) {
    if (length(item) == 1) {
      return(self$.getBatchSingle(item))
    } else {
      return(self$.getBatchRegular(item))
    }
  },
  .getBatchSingle = function(item) {
    # add leading singleton dimension since models expects 2d tensors
    batch <- list(
      cat = self$cat[item]$unsqueeze(1),
      num = self$num[item]$unsqueeze(1)
    )
    return(list(
      batch = batch,
      target = self$target[item]$unsqueeze(1)
    ))
  },
  .getBatchRegular = function(item) {
    batch <- list(
      cat = self$cat[item],
      num = self$num[item]
    )
    return(list(
      batch = batch,
      target = self$target[item]
    ))
  },
  .length = function() {
    self$target$size()[[1]] # shape[1]
  }
)
