Dataset <- torch::dataset(
  name = 'Dataset',
  
  initialize = function(data, labels, indices, numericalIndex) {
    
    self$collate_fn <- sparseCollate  
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
    return(list(cat = self$cat[item], 
                num = self$num[item,],
                target = self$target[item]))
  },
  
  .length = function() {
    self$target$size()[[1]] # shape[1]
  }
)

# a function to speed up the collation so I dont' call to_dense()
# on the sparse tensors until they have been combined for the batch
sparseCollate <- function(batch) {
  browser()
  elem <- batch[[1]]
  if (inherits(elem, "torch_tensor")) {
    # temporary fix using a tryCatch until torch in R author adds
    # an is_sparse method or exposes tensor&layout
    tryCatch(return(torch::torch_stack(batch,dim = 1)$to_dense()),
             error=function(e) return(torch::torch_stack(batch, dim = 1)))  
      
  # if (Reduce("*", elem$size()) > elem$numel()) {
  #     return(torch::torch_stack(batch,dim = 1)$to_dense())
  #   }
  #   
  #   return(torch::torch_stack(batch, dim = 1))
  }
  else if (is.list(elem)) {
   
    # preserve names of elements 
    named_seq <- seq_along(elem)
    names(named_seq) <- names(elem)
    
    lapply(named_seq, function(i) {sparseCollate(lapply(batch, function(x) x[[i]]))})
  }
}