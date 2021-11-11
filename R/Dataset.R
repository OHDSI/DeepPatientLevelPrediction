Dataset <- torch::dataset(
  name = 'Dataset',
  
  initialize=function(data, labels, indices, numericalIndex) {
    
    # add labels
    self$target <- torch::torch_tensor(labels[indices])
    
    # add features
    #print(dim(as.matrix(data[indices,]))) ## testing
    # TODO should be torch sparse COO matrix
    self$cat <- torch::torch_tensor(as.matrix(data[indices,-numericalIndex, drop = F]), dtype=torch::torch_float32())
    self$num <- torch::torch_tensor(as.matrix(data[indices,numericalIndex, drop = F]), dtype=torch::torch_float32())
    
  },
  
  .getitem = function(item) {
    return(list(cat = self$cat[item,], 
                num = self$num[item,],
                target = self$target[item]))
  },
  
  .length = function() {
    self$target$size()[[1]] # shape[1]
  }
)