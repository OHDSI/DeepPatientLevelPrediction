source('R/Formatting.R')

#' Convert the plpData in COO format into a sparse Torch tensor
#'
#' @description
#' Converts the standard plpData to a sparse tensor for Torch
#'
#' @details
#' This function converts the covariate file from COO format into a sparse Torch tensor
#' @param plpData                       An object of type \code{plpData} with covariate in coo format - the patient level prediction
#'                                      data extracted from the CDM.
#' @param population                    The population to include in the matrix
#' @param map                           A covariate map (telling us the column number for covariates)
#' @param temporal                      Whether you want to convert temporal data
#' @examples
#' #TODO
#'
#' @return
#' Returns a list, containing the data as a sparse matrix, the plpData covariateRef
#' and a data.frame named map that tells us what covariate corresponds to each column
#' This object is a list with the following components: \describe{
#' \item{data}{A sparse matrix with the rows corresponding to each person in the plpData and the columns corresponding to the covariates.}
#' \item{covariateRef}{The plpData covariateRef.}
#' \item{map}{A data.frame containing the data column ids and the corresponding covariateId from covariateRef.}
#' }
#'
#' @export
toSparseRTorch <- function(plpData, population, map=NULL, temporal=T){
  
  newCovariateData <- MapCovariates(plpData$covariateData,
                                    population, 
                                    mapping=map)

  if(temporal){
    indicesTemporal <- newCovariateData$covariates %>% filter(!is.na(.data$timeId)) %>% mutate(timeId = .data$timeId+1) %>% select(.data$rowId, .data$covariateId, .data$timeId) %>% collect() %>% as.matrix() 
    valuesTemporal <- newCovariateData$covariates %>% filter(!is.na(.data$timeId)) %>% select(.data$covariateValue) %>% collect() %>% as.matrix() 
    
    indicesTensor <- torch::torch_tensor(indicesTemporal, dtype=torch::torch_long())$t() 
    valuesTensor <- torch::torch_tensor(valuesTemporal)$squeeze()
    
    sparseMatrixTemporal <- torch::torch_sparse_coo_tensor(indices=indicesTensor,
                                                           values=valuesTensor) 
    
    indicesNonTemporal <- newCovariateData$covariates %>% filter(is.na(.data$timeId)) %>% select(.data$rowId, .data$covariateId) %>% collect() %>% as.matrix()
    valuesNonTemporal <- newCovariateData$covariates %>% filter(is.na(.data$timeId)) %>% select(.data$covariateValue) %>% collect() %>% as.matrix() 
    
  } else{
    sparseMatrixTemporal <- NULL
    indicesNonTemporal <- newCovariateData$covariates %>% select(.data$rowId, .data$covariateId) %>% collect() %>% as.matrix()
    valuesNonTemporal <- newCovariateData$covariates %>% select(.data$covariateValue) %>% collect() %>% as.matrix() 
    }
  
  indicesTensor <- torch::torch_tensor(indicesNonTemporal, dtype=torch::torch_long())$t() 
  valuesTensor <- torch::torch_tensor(valuesNonTemporal)$squeeze()
  sparseMatrixNonTemporal <- torch::torch_sparse_coo_tensor(indices=indicesTensor,
                                                 values=valuesTensor)  
  results = list(
    data = sparseMatrixNonTemporal,
    dataTemporal = sparseMatrixTemporal,
    covariateRef=as.data.frame(newCovariateData$covariateRef),
    map=as.data.frame(newCovariateData$mapping))
  
  return(results)
  
}