Dataset <- torch::dataset(
  name = 'Dataset',
  
  initialize=function(data, labels, indices, numericalIndex) {
    
    # add labels
    self$target <- torch::torch_tensor(labels[indices])
    
    # add features
    dataCat <- data[indices,-numericalIndex]
    matrix <- as(dataCat, 'dgTMatrix') # convert to triplet sparse format
    sparseIndices <- torch::torch_tensor(matrix(c(matrix@i + 1, matrix@j + 1), ncol=2), dtype = torch::torch_long())
    values <- torch::torch_tensor(matrix(c(matrix@x)), dtype = torch::torch_float32())
    self$cat <- torch::torch_sparse_coo_tensor(indices=sparseIndices$t(), 
                                               values=values$squeeze(), 
                                               dtype=torch::torch_float32())$coalesce()
    self$num <- torch::torch_tensor(as.matrix(data[indices,numericalIndex, drop = F]), dtype=torch::torch_float32())
    
  },
  
  .getitem = function(item) {
    return(list(cat = self$cat[item]$to_dense(), 
                num = self$num[item,],
                target = self$target[item]))
  },
  
  .length = function() {
    self$target$size()[[1]] # shape[1]
  }
)