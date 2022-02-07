#' @export
Dataset_plp5 <- torch::dataset(
  name = 'Dataset',
  initialize = function(data, labels = NULL, numericalIndex = NULL) {
    # determine numeric
    if(is.null(numericalIndex)){
      colBin <- apply(data, 2, function(x) sum(x==1 | x==0))
      colLen <- apply(data, 2, length)
      numericalIndex <- colLen != colBin
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
    self$all <- torch::torch_tensor(as.matrix(data), dtype = torch::torch_float32())
    
    # add features
    dataCat <- data[, !numericalIndex]
    self$cat <- torch::torch_tensor(as.matrix(dataCat), dtype=torch::torch_float32())
    
    # comment out the sparse matrix for now, is really slow need to find 
    # a better solution for converting it to dense before feeding to model
    # matrix <- as(dataCat, 'dgTMatrix') # convert to triplet sparse format
    # sparseIndices <- torch::torch_tensor(matrix(c(matrix@i + 1, matrix@j + 1), ncol=2), dtype = torch::torch_long())
    # values <- torch::torch_tensor(matrix(c(matrix@x)), dtype = torch::torch_float32())
    # self$cat <- torch::torch_sparse_coo_tensor(indices=sparseIndices$t(), 
    #                                            values=values$squeeze(), 
    #                                            dtype=torch::torch_float32())$coalesce()
    self$num <- torch::torch_tensor(as.matrix(data[,numericalIndex, drop = F]), dtype=torch::torch_float32())
  },
  
  .getNumericalIndex = function() {
    return(
      self$numericalIndex
    )
  },
  
  .getbatch = function(item) {
    return(list(cat = self$cat[item],
                num = self$num[item],
                target = self$target[item]))
  },
  
  .length = function() {
    self$target$size()[[1]] # shape[1]
  }
)


Dataset <- torch::dataset(
  name = 'Dataset',
  
  initialize = function(data, labels, numericalIndex) {
    
    # self$collate_fn <- sparseCollate  
    # add labels
    self$target <- torch::torch_tensor(labels)
    
    # Weight to add in loss function to positive class
    self$posWeight <- ((self$target==0)$sum()/self$target$sum())$item()
    
    # add features
    dataCat <- data[,-numericalIndex]
    self$cat <- torch::torch_tensor(as.matrix(dataCat), dtype=torch::torch_float32())
    
            
    # comment out the sparse matrix for now, is really slow need to find 
    # a better solution for converting it to dense before feeding to model
    # matrix <- as(dataCat, 'dgTMatrix') # convert to triplet sparse format
    # sparseIndices <- torch::torch_tensor(matrix(c(matrix@i + 1, matrix@j + 1), ncol=2), dtype = torch::torch_long())
    # values <- torch::torch_tensor(matrix(c(matrix@x)), dtype = torch::torch_float32())
    # self$cat <- torch::torch_sparse_coo_tensor(indices=sparseIndices$t(),
    #                                            values=values$squeeze(),
    #                                            dtype=torch::torch_float32())$coalesce()
    self$num <- torch::torch_tensor(as.matrix(data[,numericalIndex, drop = F]), dtype=torch::torch_float32())
  },
  .getbatch = function(item) {
    return(list(cat = self$cat[item],
                num = self$num[item],
                target = self$target[item]))
  },
  .length = function() {
    length(self$target) # shape[1]
  }
)

# a function to speed up the collation so I dont' call to_dense()
# on the sparse tensors until they have been combined for the batch
sparseCollate <- function(batch) {
  elem <- batch[[1]]
  if (inherits(elem, "torch_tensor")) {
    # temporary fix using a tryCatch until torch in R author adds
    # an is_sparse method or exposes tensor$layout
    return (torch::torch_stack(batch, dim = 1))
    # tryCatch(return(torch::torch_stack(batch,dim = 1)$to_dense()),
    #          error=function(e) return(torch::torch_stack(batch, dim = 1)))  
  }
  else if (is.list(elem)) {
    # preserve names of elements 
    named_seq <- seq_along(elem)
    names(named_seq) <- names(elem)
    
    lapply(named_seq, function(i) {sparseCollate(lapply(batch, function(x) x[[i]]))})
  }
}








