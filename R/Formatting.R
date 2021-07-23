# @file formatting.R
#
# Copyright 2020 Observational Health Data Sciences and Informatics
#
# This file is part of PatientLevelPrediction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#' Convert the plpData in COO format into a sparse R matrix
#'
#' @description
#' Converts the standard plpData to a sparse matrix
#'
#' @details
#' This function converts the covariate file from ffdf in COO format into a sparse matrix from
#' the package Matrix
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
toSparseMDeep <- function(plpData,
                          population, 
                          map=NULL, 
                          temporal=F){
  # check logger
  ParallelLogger::logInfo(paste0('starting toSparseM'))
  ParallelLogger::logDebug(paste0('covariates nrow: ', nrow(plpData$covariateData$covariates)))
  ParallelLogger::logDebug(paste0('covariateRef nrow: ', nrow(plpData$covariateData$covariateRef)))
  
   
  #assign newIds to covariateRef
  newcovariateData <- MapCovariates(plpData$covariateData,
                                 population, 
                                 mapping=map)
  
  ParallelLogger::logDebug(paste0('Max covariateId in covariates: ',as.data.frame(newcovariateData$covariates %>% dplyr::summarise(max = max(.data$covariateId, na.rm=T)))))
  ParallelLogger::logDebug(paste0('# covariates in covariateRef: ', nrow(newcovariateData$covariateRef)))
  ParallelLogger::logDebug(paste0('Max rowId in covariates: ', as.data.frame(newcovariateData$covariates %>% dplyr::summarise(max = max(.data$rowId, na.rm=T)))))

  maxY <- as.data.frame(newcovariateData$mapping %>% dplyr::summarise(max=max(.data$newCovariateId, na.rm = TRUE)))$max
  ParallelLogger::logDebug(paste0('Max newCovariateId in mapping: ',maxY))
  maxX <- max(population$rowId)
  ParallelLogger::logDebug(paste0('Max rowId in population: ',maxX))
  
  # chunk then add
  if(!temporal){
    
    ParallelLogger::logInfo(paste0('Casting plpData into vectors and creating sparseMatrix - you need suffciently large RAM for this '))
    ParallelLogger::logInfo(paste0('toSparseMDeep non temporal used'))
    
    data <- Matrix::sparseMatrix(i=as.data.frame(newcovariateData$covariates %>% dplyr::select(.data$rowId))$rowId,
                                        j=as.data.frame(newcovariateData$covariates %>% dplyr::select(.data$covariateId))$covariateId,
                                        x=as.data.frame(newcovariateData$covariates %>% dplyr::select(.data$covariateValue))$covariateValue,
                                        dims=c(maxX,maxY))
  } else {
    ParallelLogger::logInfo(paste0('toSparseMDeep temporal used'))
    
    minT <- min(newcovariateData$covariates %>% dplyr::select(.data$timeId) %>% collect(), na.rm = T)
    maxT <- max(newcovariateData$covariates %>% dplyr::select(.data$timeId) %>% collect(), na.rm = T)
    
    ParallelLogger::logTrace(paste0('Min time:', minT))
    ParallelLogger::logTrace(paste0('Max time:', maxT))
    
    # do we want to use for(i in sort(plpData$timeRef$timeId)){ ?
    for(i in minT:maxT){
      
      if(newcovariateData$covariates %>% dplyr::filter(.data$timeId==i) %>% dplyr::summarise(n=n()) %>% dplyr::collect() > 0  ){
        ParallelLogger::logInfo(paste0('Found covariates for timeId ', i))

        dataPlp <- Matrix::sparseMatrix(i= as.integer(as.character(as.data.frame(newcovariateData$covariates  %>% dplyr::filter(.data$timeId==i) %>% dplyr::select(.data$rowId))$rowId)),
                                        j= as.integer(as.character(as.data.frame(newcovariateData$covariates  %>% dplyr::filter(.data$timeId==i) %>% dplyr::select(.data$covariateId))$covariateId)),
                                        x= as.double(as.character(as.data.frame(newcovariateData$covariates  %>% dplyr::filter(.data$timeId==i) %>% dplyr::select(.data$covariateValue))$covariateValue)),
                                        dims=c(maxX,maxY))
        
        data_array <- slam::as.simple_sparse_array(dataPlp)
        #extending one more dimesion to the array
        data_array <- slam::extend_simple_sparse_array(data_array, MARGIN =c(1L))
        ParallelLogger::logInfo(paste0('Finished Mapping covariates for timeId ', i))
 
      } else {
        data_array <- tryCatch(slam::simple_sparse_array(i=matrix(c(1,1,1), ncol = 3), 
                                                         v=0,
                                                         dim=c(maxX,1, maxY))
        )
        
      }
      
      # add na timeIds - how?
      
       #binding arrays along the dimesion
      if(i==minT) {
        result_array <- data_array
      }else{
        result_array <- slam::abind_simple_sparse_array(result_array,data_array,MARGIN=2L)
      }
    }
    data <- result_array
  }
  
  ParallelLogger::logDebug(paste0('Sparse matrix with dimensionality: ', paste(dim(data), collapse=',')  ))

  ParallelLogger::logInfo(paste0('finishing toSparseM'))
  
  result <- list(data=data,
                 covariateRef=as.data.frame(newcovariateData$covariateRef),
                 map=as.data.frame(newcovariateData$mapping))
  return(result)

}

# restricts to pop and saves/creates mapping
MapCovariates <- function(covariateData,population, mapping=NULL){
  
  # to remove check notes
  #covariateId <- oldCovariateId <- newCovariateId <- NULL
  ParallelLogger::logInfo(paste0('starting MapCovariates'))
  
  newCovariateData <- Andromeda::andromeda(covariateRef = covariateData$covariateRef,
                                           analysisRef = covariateData$analysisRef)
  
  # restrict to population for speed
  ParallelLogger::logTrace('restricting to population for speed and mapping')
  if(is.null(mapping)){
    mapping <- data.frame(oldCovariateId = as.data.frame(covariateData$covariateRef %>% dplyr::select(.data$covariateId)),
                          newCovariateId = 1:nrow(as.data.frame(covariateData$covariateRef)))
  }
  if(sum(colnames(mapping)%in%c('oldCovariateId','newCovariateId'))!=2){
    colnames(mapping) <- c('oldCovariateId','newCovariateId')
  }
  covariateData$mapping <- mapping
  covariateData$population <- data.frame(rowId = population[,'rowId'])
  # assign new ids :
  newCovariateData$covariates <- covariateData$covariates %>%
    dplyr::inner_join(covariateData$population) %>% 
    dplyr::rename(oldCovariateId = .data$covariateId) %>% 
    dplyr::inner_join(covariateData$mapping) %>% 
    dplyr::select(- .data$oldCovariateId)  %>%
    dplyr::rename(covariateId = .data$newCovariateId)
  covariateData$population <- NULL
  covariateData$mapping <- NULL
  
  newCovariateData$mapping <- mapping
  
  ParallelLogger::logInfo(paste0('finished MapCovariates'))
  
  return(newCovariateData)
}

