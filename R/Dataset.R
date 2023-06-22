#' #'  A torch dataset
#' #' @export
#' Dataset <- torch::dataset(
#'   name = "myDataset",
#'   #' @param data           a dataframe like object with the covariates
#'   #' @param labels         a dataframe with the labels
#'   #' @param numericalIndex in what column numeric data is in (if any)
#'   initialize = function(data, labels = NULL, numericalIndex = NULL) {
#'     # determine numeric
#'     if (is.null(numericalIndex)) {
#'       numericalIndex <- data %>%
#'         dplyr::arrange(columnId) %>% 
#'         dplyr::group_by(columnId) %>%
#'         dplyr::summarise(n = dplyr::n_distinct(.data$covariateValue)) %>%
#'         dplyr::collect() %>%
#'         dplyr::pull(n) > 1
#'       self$numericalIndex <- numericalIndex
#'     } else {
#'       self$numericalIndex <- NULL
#'     }
#' 
#'     # add labels if training (make 0 vector for prediction)
#'     if (!is.null(labels)) {
#'       self$target <- torch::torch_tensor(labels)
#'     } else {
#'       self$target <- torch::torch_tensor(rep(0, data %>% dplyr::summarize(m=max(rowId)) %>%
#'                                                dplyr::collect() %>% dplyr::pull()))
#'     }
#'     # Weight to add in loss function to positive class
#'     self$posWeight <- (self$target == 0)$sum() / self$target$sum()
#' 
#'     # add features
#'     catColumns <- which(!numericalIndex)
#'     dataCat <- dplyr::filter(data, columnId %in% catColumns) %>%
#'       dplyr::arrange(columnId) %>%
#'       dplyr::group_by(columnId) %>%
#'       dplyr::collect() %>%
#'       dplyr::mutate(newColumnId = dplyr::cur_group_id()) %>%
#'       dplyr::ungroup() %>%
#'       dplyr::select(c("rowId", "newColumnId")) %>%
#'       dplyr::rename(columnId = newColumnId) %>% 
#'       dplyr::arrange(rowId)
#'     start <- Sys.time()
#'     catTensor <- torch::torch_tensor(cbind(dataCat$rowId, dataCat$columnId))
#'     catTensor <- catTensor[catTensor[,1]$argsort(),]
#'     tensorList <- torch::torch_split(catTensor[,2], 
#'                                      as.numeric(torch::torch_unique_consecutive(catTensor[,1], 
#'                                                                                 return_counts = TRUE)[[3]]))
#'     
#'     # because of subjects without cat features, I need to create a list with all zeroes and then insert
#'     # my tensorList. That way I can still index the dataset correctly.
#'     totalList <- as.list(integer(length(self$target)))
#'     totalList[unique(dataCat$rowId)] <- tensorList
#'     self$lengths <- lengths
#'     self$cat <- torch::nn_utils_rnn_pad_sequence(totalList, batch_first = T)
#'     delta <- Sys.time() - start
#'     ParallelLogger::logInfo("Data conversion for dataset took ", signif(delta, 3), " ", attr(delta, "units"))
#'     if (sum(numericalIndex) == 0) {
#'       self$num <- NULL
#'     } else {
#'       numericalData <- data %>%
#'         dplyr::filter(columnId %in% !!which(numericalIndex)) %>%
#'         dplyr::collect()
#'       numericalData <- numericalData %>%
#'         dplyr::group_by(columnId) %>%
#'         dplyr::mutate(newId = dplyr::cur_group_id())
#'       indices <- torch::torch_tensor(cbind(numericalData$rowId, numericalData$newId), dtype = torch::torch_long())$t_()
#'       values <- torch::torch_tensor(numericalData$covariateValue, dtype = torch::torch_float32())
#'       self$num <- torch::torch_sparse_coo_tensor(
#'         indices = indices,
#'         values = values,
#'         size = c(self$target$shape, sum(numericalIndex))
#'       )$to_dense()
#'     }
#'   },
#'   getNumericalIndex = function() {
#'     return(
#'       self$numericalIndex
#'     )
#'   },
#'   numCatFeatures = function() {
#'     return(
#'       sum(!self$numericalIndex)
#'     )
#'   },
#'   numNumFeatures = function() {
#'     if (!is.null(self$num)) {
#'       return(self$num$shape[2])
#'     } else {
#'       return(0)
#'     }
#'   },
#'   .getbatch = function(item) {
#'     if (length(item) == 1) {
#'       return(self$.getBatchSingle(item))
#'     } else {
#'       return(self$.getBatchRegular(item))
#'     }
#'   },
#'   .getBatchSingle = function(item) {
#'     # add leading singleton dimension since models expects 2d tensors
#'     batch <- list(
#'       cat = self$cat[item]$unsqueeze(1),
#'       num = self$num[item]$unsqueeze(1)
#'     )
#'     return(list(
#'       batch = batch,
#'       target = self$target[item]$unsqueeze(1)
#'     ))
#'   },
#'   .getBatchRegular = function(item) {
#'     batch <- list(
#'       cat = self$cat[item],
#'       num = self$num[item]
#'     )
#'     return(list(
#'       batch = batch,
#'       target = self$target[item]
#'     ))
#'   },
#'   .length = function() {
#'     self$target$size()[[1]] # shape[1]
#'   }
#' )
