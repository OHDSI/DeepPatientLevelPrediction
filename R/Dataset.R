#'  A torch dataset 
#' @import data.table
#' @export
Dataset <- torch::dataset(
  name = 'myDataset',
  #' @param data           a dgCSparseMatrix with the features
  #' @param labels         a dataframe with the labels
  #' @param numericalIndex in what column numeric data is in (if any)
  initialize = function(data, labels = NULL, numericalIndex = NULL) {
    # determine numeric
    if(is.null(numericalIndex)){
      colList <- listCols(data)
      numericalIndex <- vapply(colList, function(x) sum(x==1 | x==0) != length(x), 
                               TRUE)
    }
    
    self$numericalIndex <- numericalIndex
    
    self$collate_fn <- sparseCollate  
    # add labels if training (make 0 vector for prediction)
    if(!is.null(labels)){
      self$target <- torch::torch_tensor(labels)
    } else{
      self$target <- torch::torch_tensor(rep(0, nrow(data)))
    }
    
    # Weight to add in loss function to positive class
    self$posWeight <- ((self$target==0)$sum()/self$target$sum())$item()
    # for DeepNNTorch
    # self$all <- torch::torch_tensor(as.matrix(data), dtype = torch::torch_float32())
  
    # add features
    dataCat <- data[, !numericalIndex]
    dataCat <- as(dataCat, 'dgTMatrix')
    
    # the fastest way I found so far to convert data using data.table
    # 1.5 min for 100k rows :(
    dt <- data.table::data.table(rows=dataCat@i+1L, cols=dataCat@j+1L)
    maxFeatures <- max(dt[, .N, by=rows][,N])
    start <- Sys.time()
    cat <- lapply(1:dim(dataCat)[[1]], function(x) {
      currRow <- dt[rows==x, cols]
      maxCols <- length(currRow)
      torch::torch_cat(list(torch::torch_tensor(currRow),torch::torch_zeros((maxFeatures - maxCols),
                            dtype=torch::torch_long()))
                       )
      })
    self$lengths <- lengths
    self$cat <- torch::torch_vstack(cat)
    delta <- Sys.time() - start
    ParallelLogger::logInfo("Data conversion for dataset took ", signif(delta, 3), " ", attr(delta, "units"))
    
    if (sum(numericalIndex) == 0) {
      self$num <- NULL
    } else  {
      self$num <- torch::torch_tensor(as.matrix(data[,numericalIndex, drop = F]), dtype=torch::torch_float32())
    } 
  },
  
  getNumericalIndex = function() {
    return(
      self$numericalIndex
    )
  },
  
  numCatFeatures = function() {
    return (
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
    if (length(item)==1) {
      # add leading singleton dimension since models expects 2d tensors
      return(list(batch = list(cat = self$cat[item],
                               num = self$num[item]),
                  target = self$target[item]$unsqueeze(1)))
      }
    else {
    return(list(batch = list(cat = self$cat[item],
                             num = self$num[item]),
                target = self$target[item]))}
  },
  
  .length = function() {
    self$target$size()[[1]] # shape[1]
  }
)

#' @description converts a sparse Matrix into a list of its columns, 
#' subsequently vapply can be used to apply functions over the list 
listCols<-function(m){
  res<-split(m@x, findInterval(seq_len(length(m@x)), m@p, left.open=TRUE))
  names(res)<-colnames(m)
  res
}





