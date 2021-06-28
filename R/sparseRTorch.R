source('R/Formatting.R')

toSparseRTorch <- function(plpData, population, map=NULL, temporal=T){
  
  newCovariateData <- MapCovariates(plpData$covariateData,
                                    population, 
                                    mapping=map)

  indices <- newCovariateData$covariates %>% select(rowId, covariateId, timeId) %>% collect() %>% as.matrix()
  values <- newCovariateData$covariates %>% select(covariateValue) %>% collect() %>% as.matrix()
  
  indicesTensor <- torch::torch_tensor(indices, dtype=torch::torch_long())$t() 
  valuesTensor <- torch::torch_tensor(values)$squeeze()
  
  sparseMatrix <- torch::torch_sparse_coo_tensor(indices=indicesTensor,
                                                 values=valuesTensor)  
  results = list(
    data=sparseMatrix,
    covariateRef=as.data.frame(newCovariateData$covariateRef),
    map=as.data.frame(newCovariateData$mapping))
  
  return(results)
  
}