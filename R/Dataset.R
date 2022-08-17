#'  A torch dataset
#' @import data.table
#' @export
Dataset <- torch::dataset(
  name = "myDataset",
  #' @param data           a dataframe like object with the covariates
  #' @param labels         a dataframe with the labels
  #' @param numericalIndex in what column numeric data is in (if any)
  #' @param all            if True then returns all features instead of splitting num/cat
  initialize = function(data, labels = NULL, numericalIndex = NULL, all = FALSE) {
    # determine numeric
    if (is.null(numericalIndex) && all == FALSE) {
      numericalIndex <- data %>%
        dplyr::group_by(columnId) %>%
        dplyr::collect() %>%
        dplyr::summarise(n = dplyr::n_distinct(.data$covariateValue)) %>%
        dplyr::pull(n) > 1
      self$numericalIndex <- numericalIndex
    } else {
      self$numericalIndex <- NULL
    }


    # add labels if training (make 0 vector for prediction)
    if (!is.null(labels)) {
      self$target <- torch::torch_tensor(labels)
    } else {
      if (all == FALSE) {
        self$target <- torch::torch_tensor(rep(0, data %>% dplyr::distinct(rowId)
          %>% dplyr::collect() %>% nrow()))
      } else {
        self$target <- torch::torch_tensor(rep(0, dim(data)[[1]]))
      }
    }
    # Weight to add in loss function to positive class
    self$posWeight <- (self$target == 0)$sum() / self$target$sum()
    # for DeepNNTorch
    if (all) {
      self$all <- torch::torch_tensor(as.matrix(data), dtype = torch::torch_float32())
      self$cat <- NULL
      self$num <- NULL
      return()
    }
    # add features
    catColumns <- which(!numericalIndex)
    dataCat <- dplyr::filter(data, columnId %in% catColumns) %>%
      dplyr::arrange(columnId) %>%
      dplyr::group_by(columnId) %>%
      dplyr::collect() %>%
      dplyr::mutate(newColumnId = dplyr::cur_group_id()) %>%
      dplyr::ungroup() %>%
      dplyr::select(c("rowId", "newColumnId")) %>%
      dplyr::rename(columnId = newColumnId)
    # the fastest way I found so far to convert data using data.table
    # 1.5 min for 100k rows :(
    dt <- data.table::data.table(rows = dataCat$rowId, cols = dataCat$columnId)
    maxFeatures <- max(dt[, .N, by = rows][, N])
    start <- Sys.time()
    tensorList <- lapply(1:max(data %>% dplyr::pull(rowId)), function(x) {
      torch::torch_tensor(dt[rows == x, cols])
    })
    self$lengths <- lengths
    self$cat <- torch::nn_utils_rnn_pad_sequence(tensorList, batch_first = T)
    delta <- Sys.time() - start
    ParallelLogger::logInfo("Data conversion for dataset took ", signif(delta, 3), " ", attr(delta, "units"))
    if (sum(numericalIndex) == 0) {
      self$num <- NULL
    } else {
      numericalData <- data %>%
        dplyr::filter(columnId %in% !!which(numericalIndex)) %>%
        dplyr::collect()
      numericalData <- numericalData %>%
        dplyr::group_by(columnId) %>%
        dplyr::mutate(newId = dplyr::cur_group_id())
      indices <- torch::torch_tensor(cbind(numericalData$rowId, numericalData$newId), dtype = torch::torch_long())$t_()
      values <- torch::torch_tensor(numericalData$covariateValue, dtype = torch::torch_float32())
      self$num <- torch::torch_sparse_coo_tensor(
        indices = indices,
        values = values,
        size = c(self$target$shape, sum(numericalIndex))
      )$to_dense()
    }
    if (self$cat$shape[1] != self$num$shape[1]) {
      browser()
    }
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
      # add leading singleton dimension since models expects 2d tensors
      return(list(
        batch = list(
          cat = self$cat[item]$unsqueeze(1),
          num = self$num[item]$unsqueeze(1)
        ),
        target = self$target[item]$unsqueeze(1)
      ))
    } else {
      return(list(
        batch = list(
          cat = self$cat[item],
          num = self$num[item]
        ),
        target = self$target[item]
      ))
    }
  },
  .length = function() {
    self$target$size()[[1]] # shape[1]
  }
)
