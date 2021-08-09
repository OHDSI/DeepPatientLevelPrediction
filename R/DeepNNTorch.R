# @file DeepNNTorch.R
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

#' Create setting for DeepNN model using Torch for R
#'
#' @param units         The number of units of the deep network - as a list of vectors
#' @param layer_dropout      The layer dropout rate (regularisation)
#' @param lr                 Learning rate
#' @param decay              Learning rate decay over each update.
#' @param outcome_weight      The weight of the outcome class in the loss function
#' @param batch_size          The number of data points to use per training batch
#' @param epochs          Number of times to iterate over dataset
#' @param seed            Random seed used by deep learning model
#'
#' @examples
#' \dontrun{
#' model <- setDeepNN()
#' }
#' @export
setDeepNNTorch <- function(units=list(c(128, 64), 128), layer_dropout=c(0.2),
                      lr =c(1e-4), decay=c(1e-5), outcome_weight = c(1.0), batch_size = c(100), 
                      epochs= c(100),  seed=NULL  ){
  
  # ensure_installed("torch")
  
  # if(class(indexFolder)!='character')
  #     stop('IndexFolder must be a character')
  # if(length(indexFolder)>1)
  #     stop('IndexFolder must be one')
  # 
  # if(class(units)!='numeric')
  #     stop('units must be a numeric value >0 ')
  # if(units<1)
  #     stop('units must be a numeric value >0 ')
  # 
  # #if(length(units)>1)
  # #    stop('units can only be a single value')
  # 
  # if(class(recurrent_dropout)!='numeric')
  #     stop('dropout must be a numeric value >=0 and <1')
  # if( (recurrent_dropout<0) | (recurrent_dropout>=1))
  #     stop('dropout must be a numeric value >=0 and <1')
  # if(class(layer_dropout)!='numeric')
  #     stop('layer_dropout must be a numeric value >=0 and <1')
  # if( (layer_dropout<0) | (layer_dropout>=1))
  #     stop('layer_dropout must be a numeric value >=0 and <1')
  # if(class(lr)!='numeric')
  #     stop('lr must be a numeric value >0')
  # if(lr<=0)
  #     stop('lr must be a numeric value >0')
  # if(class(decay)!='numeric')
  #     stop('decay must be a numeric value >=0')
  # if(decay<=0)
  #     stop('decay must be a numeric value >=0')
  # if(class(outcome_weight)!='numeric')
  #     stop('outcome_weight must be a numeric value >=0')
  # if(outcome_weight<=0)
  #     stop('outcome_weight must be a numeric value >=0')
  # if(class(batch_size)!='numeric')
  #     stop('batch_size must be an integer')
  # if(batch_size%%1!=0)
  #     stop('batch_size must be an integer')
  # if(class(epochs)!='numeric')
  #     stop('epochs must be an integer')
  # if(epochs%%1!=0)
  #     stop('epochs must be an integer')
  # if(!class(seed)%in%c('numeric','NULL'))
  #     stop('Invalid seed')
  #if(class(UsetidyCovariateData)!='logical')
  #    stop('UsetidyCovariateData must be an TRUE or FALSE')
  
  param <- expand.grid(units=units,
                       layer_dropout=layer_dropout,
                       lr =lr, decay=decay, outcome_weight=outcome_weight,epochs= epochs,
                       seed=ifelse(is.null(seed),'NULL', seed))
  param$units1=unlist(lapply(param$units, function(x) x[1])) 
  param$units2=unlist(lapply(param$units, function(x) x[2])) 
  param$units3=unlist(lapply(param$units, function(x) x[3]))
  
  result <- list(model='fitDeepNNTorch', param=split(param,
                                                1:(length(units)*length(layer_dropout)*length(lr)*length(decay)*length(outcome_weight)*length(epochs)*max(1,length(seed)))),
                 name='DeepNNTorch'
  )
  
  class(result) <- 'modelSettings' 
  return(result)
}

#' @export
fitDeepNNTorch <- function(plpData,population, param, search='grid', quiet=F,
                      outcomeId, cohortId, ...){
  # check plpData is coo format:
  if (!FeatureExtraction::isCovariateData(plpData$covariateData)){
    stop('DeepNNTorch requires correct covariateData')
  }
  if(!is.null(plpData$timeRef)){
    warning('Data temporal but current deepNNTorch uses non-temporal data...')
    # This can be changed after supporting the temporal covariates.
  }
  if(!is.null(plpData$metaData$call$covariateSettings$temporalSequence)){
    if(plpData$metaData$call$covariateSettings$temporalSequence){
    warning('Data temporal but current deepNNTorch uses non-temporal data...')
    # This can be changed after supporting the temporal covariates.
  }}
  
  metaData <- attr(population, 'metaData')
  if(!is.null(population$indexes))
    population <- population[population$indexes>0,]
  attr(population, 'metaData') <- metaData
  
  start<-Sys.time()
  
  result<- toSparseM(plpData,population,map=NULL, temporal=F)
  data <- result$data
  
  #one-hot encoding
  y <- population$outcomeCount
  y[y>0] <- 1
  population$y <- cbind(matrix(y), matrix(abs(y-1)))
  
  
  # do cross validation to find hyperParameter
  datas <- list(population=population, plpData=data)
  hyperParamSel <- list()
  
  for(i in 1:length(param)){
    hyperParamSel[[i]] <- do.call(trainDeepNNTorch, c(param[i][[1]],datas,train = TRUE))
  }
  
  hyperSummary <- cbind(do.call(rbind, lapply(hyperParamSel, function(x) x$hyperSum)))
  hyperSummary <- as.data.frame(hyperSummary)
  hyperSummary$auc <- unlist(lapply(hyperParamSel, function (x) x$auc))
  hyperParamSel<-unlist(lapply(hyperParamSel, function(x) x$auc))
  
  #now train the final model and return coef
  bestInd <- which.max(abs(unlist(hyperParamSel)-0.5))[1]
  finalModel<-do.call(trainDeepNNTorch, c(param[bestInd][[1]],datas, train=FALSE))
  
  covariateRef <- as.data.frame(plpData$covariateData$covariateRef)
  incs <- rep(1, nrow(covariateRef)) 
  covariateRef$included <- incs
  covariateRef$covariateValue <- rep(0, nrow(covariateRef))
  
  #modelTrained <- file.path(outLoc) 
  param.best <- param[bestInd][[1]]
  
  comp <- start-Sys.time()
  
  # train prediction
  prediction <- finalModel$prediction
  finalModel$prediction <- NULL
  
  # return model location 
  result <- list(model = finalModel$model,
                 trainCVAuc = -1, # ToDo decide on how to deal with this
                 hyperParamSearch = hyperSummary,
                 modelSettings = list(model='fitDeepNN',modelParameters=param.best),
                 metaData = plpData$metaData,
                 populationSettings = attr(population, 'metaData'),
                 outcomeId=outcomeId,
                 cohortId=cohortId,
                 varImp = covariateRef, 
                 trainingTime =comp,
                 covariateMap=result$map,
                 predictionTrain = prediction
  )
  class(result) <- 'plpModel'
  attr(result, 'type') <- 'deepNNTorch'
  attr(result, 'predictionType') <- 'binary'
  
  return(result)
}

trainDeepNNTorch <-function(plpData, population,
                      units1=128, units2= NA, units3=NA, 
                      layer_dropout=0.2,
                      lr =1e-4, decay=1e-5, outcome_weight = 1.0, batch_size = 10000, 
                      epochs= 100, seed=NULL, train=TRUE,...){
  
  if(!is.null(population$indexes) && train==T){
    index_vect <- unique(population$indexes)
    ParallelLogger::logInfo(paste('Training deep neural network using Torch with ',length(index_vect ),' fold CV'))
    
    perform <- c()
    
    # create prediction matrix to store all predictions
    predictionMat <- population
    predictionMat$value <- 0
    attr(predictionMat, "metaData") <- list(predictionType = "binary")

    
    for(index in 1:length(index_vect)){
      ParallelLogger::logInfo(paste('Fold ',index, ' -- with ', sum(population$indexes!=index),'train rows'))
      
      if(is.na(units2)){
        model <- singleLayerNN(inputN = ncol(plpData),
                               layer1 = units1, 
                               outputN = 2, 
                               layer_dropout = layer_dropout)
        
      } else if(is.na(units3)){
        model <- doubleLayerNN(inputN = ncol(plpData),
                               layer1 = units1,
                               layer2 = units2,
                               outputN = 2, 
                               layer_dropout = layer_dropout)
      } else{
        model <- tripleLayerNN(inputN = ncol(plpData),
                               layer1 = units1,
                               layer2 = units2,
                               layer3 = units3,
                               outputN = 2, 
                               layer_dropout = layer_dropout)
      }

      # get the rowIds for the train/test/earlyStopping
      rowIdSet <- rowIdSets(population, index)
      
      criterion = torch::nn_bce_loss() #Binary crossentropy only
      optimizer = torch::optim_adam(model$parameters, lr = lr)
      
      # Need earlyStopping
      # Need setting decay
      
      # create batch sets
      batches <- split(rowIdSet$trainRowIds, ceiling(seq_along(rowIdSet$trainRowIds)/batch_size))
      
      for(i in 1:epochs){
        for(batchRowIds in batches){
          trainDataBatch <- convertToTorchData(plpData, 
                                               population$y, 
                                               rowIds = batchRowIds)
          
        optimizer$zero_grad()
        y_pred = model(trainDataBatch$x)
        loss = criterion(y_pred, trainDataBatch$y)
        loss$backward()
        optimizer$step()
        
        if(i%%10 == 0){
          # winners = y_pred$argmax(dim = 2) + 1
          # winners = y_pred
          # corrects = (winners = y_train)
          # accuracy = corrects$sum()$item() / y_train$size()[1]
          # cat("Epoch:", i, "Loss:", loss$item(), " Accuracy:", accuracy, "\n")
          
          cat("Epoch:", i, "Loss:", loss$item(), "\n")
          
        }
        }
      }
      
      model$eval()
      
      # batch predict
      prediction <- batchPredict(model, 
                                 plpData,
                                 population,
                                 predictRowIds = rowIdSet$testRowIds,
                                 batch_size )
      
      aucVal <- computeAuc(prediction)
      perform <- c(perform,aucVal)
      
      predictionMat <- updatePredictionMat(predictionMat,
                                           prediction)
    }
    
    auc <- computeAuc(predictionMat)
    foldPerm <- perform
    
    # Output  ----------------------------------------------------------------
    param.val <- paste('units1: ',units1,'units2: ',units2,'units3: ',units3,
                        'layer_dropout: ',layer_dropout,'-- lr: ', lr,
                        '-- decay: ', decay, '-- batch_size: ',batch_size, '-- epochs: ', epochs)
    ParallelLogger::logInfo('==========================================')
    ParallelLogger::logInfo(paste0('DeepNNTorch with parameters:', param.val,' obtained an AUC of ',auc))
    ParallelLogger::logInfo('==========================================')
    
  } else{
    
    if(is.na(units2)){
      model <- singleLayerNN(inputN = ncol(plpData),
                             layer1 = units1, 
                             outputN = 2, 
                             layer_dropout = layer_dropout)
      
    } else if(is.na(units3)){
      model <- doubleLayerNN(inputN = ncol(plpData),
                             layer1 = units1,
                             layer2 = units2,
                             outputN = 2, 
                             layer_dropout = layer_dropout)
    } else{
      model <- tripleLayerNN(inputN = ncol(plpData),
                             layer1 = units1,
                             layer2 = units2,
                             layer3 = units3,
                             outputN = 2, 
                             layer_dropout = layer_dropout)
    }
    
    # get the rowIds for the train/earlyStopping
    rowIdSet <- rowIdSets(population, index = NULL)
    
    criterion = torch::nn_bce_loss() #Binary crossentropy only
    optimizer = torch::optim_adam(model$parameters, lr = lr)
    
    # create batch sets
    batches <- split(rowIdSet$trainRowIds, ceiling(seq_along(rowIdSet$trainRowIds)/batch_size))
    
    for(i in 1:epochs){
      for(batchRowIds in batches){
        trainDataBatch <- convertToTorchData(plpData, 
                                             population$y, 
                                             rowIds = batchRowIds)
        
        optimizer$zero_grad()
        y_pred = model(trainDataBatch$x)
        loss = criterion(y_pred, trainDataBatch$y)
        loss$backward()
        optimizer$step()
        
        if(i%%10 == 0){
          cat("Epoch:", i, "Loss:", loss$item(), "\n")
          
        }
      }
    }
    model$eval()
    
    # batch predict
    prediction <- batchPredict(model, 
                               plpData,
                               population,
                               predictRowIds = population$rowId,
                               batch_size )
    
    auc <- computeAuc(prediction)
    foldPerm <- auc
    predictionMat <- prediction
  }
  
  result <- list(model=model,
                 auc=auc,
                 prediction = predictionMat,
                 hyperSum = unlist(list(units1=units1,units2=units2,units3=units3, 
                                        layer_dropout=layer_dropout,lr =lr, decay=decay,
                                        batch_size = batch_size, epochs= epochs)))
  return(result)
}
