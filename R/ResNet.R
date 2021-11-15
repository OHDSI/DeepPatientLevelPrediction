# @file ResNet.R
#
# Copyright 2021 Observational Health Data Sciences and Informatics
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

#' setResNet
#'
#' @description 
#' Creates settings for a ResNet model
#' 
#' @details
#' Model architecture from by https://arxiv.org/abs/2106.11959 
#' 
#' 
#' @param numLayers         Number of layers in network, default: 1:16
#' @param sizeHidden        Amount of neurons in each default layer, default: 2^(6:10) (64 to 1024)
#' @param hiddenFactor      How much to grow the amount of neurons in each ResLayer, default: 1:4
#' @param residualDropout   How much dropout to apply after last linear layer in ResLayer, default: seq(0, 0.3, 0.05)
#' @param hiddenDropout     How much dropout to apply after first linear layer in ResLayer, default: seq(0, 0.3, 0.05)
#' @param normalization     Which type of normalization to use. Default: 'Batchnorm'
#' @param activation        What kind of activation to use. Default: 'RelU'
#' @param sizeEmbedding     Size of embedding layer, default: 2^(6:9) (64 to 512)
#' @param weightDecay       Weight decay to apply, default: c(1e-6, 1e-3)
#' @param learningRate      Learning rate to use. default: c(1e-2, 1e-5)
#' @param seed              Seed to use for sampling hyperparameter space
#' @param hyperParamSearch  Which kind of hyperparameter search to use random sampling or exhaustive grid search. default: 'random'
#' @param randomSample      How many random samples from hyperparameter space to use
#' @param device            Which device to run analysis on, either 'cpu' or 'cuda', default: 'cpu'
#' @param batch_size        Size of batch, default: 1024
#' @param epochs            Number of epochs to run, default: 10
#'
#' @export
setResNet <- function(numLayers=1:16, sizeHidden=2^(6:10), hiddenFactor=1:4,
                      residualDropout=seq(0,0.3,0.05), hiddenDropout=seq(0,0.3,0.05),
                      normalization='BatchNorm', activation='RelU',
                      sizeEmbedding=2^(6:9), weightDecay=c(1e-6, 1e-3),
                      learningRate=c(1e-2,1e-5), seed=NULL, hyperParamSearch='random',
                      randomSample=100, device='cpu', batchSize=1024, epochs=10) {

  if (!is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  
  set.seed(seed)
  
  param <- expand.grid(numLayers=numLayers, sizeHidden=sizeHidden,
                       hiddenFactor=hiddenFactor,
                       residualDropout=residualDropout,
                       hiddenDropout=hiddenDropout,
                       sizeEmbedding=sizeEmbedding, weightDecay=weightDecay,
                       learningRate=learningRate)
  if (hyperParamSearch=='random'){
    param <- param[sample(nrow(param), randomSample),]
  }
  param$device <- device
  param$batchSize <- batchSize
  param$epochs <- epochs
  
  results <- list(model='fitResNet', param=param, name='ResNet')

  class(results) <- 'modelSettings'

  return(results)

}
#' @description 
#' fits a ResNet model to data
#' 
#' @param population    the study population dataframe
#' @param plpData       plp data object
#' @param param         parameters to use for model
#' @param outcomeId     Id of the outcome
#' @param cohortId      Id of the cohort
#' @param ... 
#'
#' @export
fitResNet <- function(population, plpData, param, outcomeId, cohortId, ...) {
  
  start <- Sys.time()
  #sparseMatrix <- toSparseM(plpData, population)
  sparseMatrix <- toSparseMDeep(plpData ,population, 
                     map=NULL, 
                     temporal=F)

  #do cross validation to find hyperParameters
  hyperParamSel <- list()
  for (i in 1:nrow(param)) {
    hyperParamSel[[i]] <- do.call(trainResNet, listAppend(param[i,], list(sparseMatrix =sparseMatrix,
                                                                           population = population,
                                                                           train=TRUE)))
  }
  hyperSummary <-as.data.frame(cbind(do.call(rbind, lapply(hyperParamSel, function(x) x$hyperSum))))
  hyperSummary$auc <- unlist(lapply(hyperParamSel, function(x) x$auc))
  
  scores <- unlist(lapply(hyperParamSel, function(x) x$auc))
  
  # now train the final model
  bestInd <- which.max(abs(unlist(scores)-0.5))[1]
  param.best <- param[bestInd,]
  uniqueEpochs <- unique(hyperSummary$bestEpochs[[bestInd]])
  param.best$epochs <- uniqueEpochs[which.max(tabulate(match(hyperSummary$bestEpochs[[bestInd]], uniqueEpochs)))]
  outLoc <- tempfile(pattern = 'resNet')
  outLoc <- file.path(outLoc, paste0('finalModel'))
  param.best$resultsDir <- outLoc
  dir.create(outLoc, recursive = TRUE)
  


  finalModel <-  do.call(trainResNet, listAppend(param.best, list(sparseMatrix = sparseMatrix,
                                                                  population = population,
                                                                  train=FALSE)))
  modelTrained <- file.path(outLoc, dir(outLoc))
  
  comp <- Sys.time() - start
  # return model location
  result <- list(model = finalModel$model,
                 trainCVAuc = scores[bestInd],
                 hyperParamSearch = hyperSummary,
                 modelSettings = list(model='fitResNet',modelParameters=param.best),
                 metaData = plpData$metaData,
                 populationSettings = attr(population, 'metaData'),
                 outcomeId=outcomeId,
                 cohortId=cohortId,
                 varImp = NULL, 
                 trainingTime =comp,
                 covariateMap=sparseMatrix$map, # I think this is need for new data to map the same?
                 predictionTrain = finalModel$prediction
  )
  class(result) <- 'plpModel'
  attr(result, 'type') <- 'deepEstimator'
  attr(result, 'predictionType') <- 'binary'
  return(result)
}

#' @param sparseMatrix 
#'
#' @param population 
#' @param ... 
#' @param train 
#'
#' @export
trainResNet <- function(sparseMatrix, population,...,train=T) {

  param <- list(...)

  modelParamNames <- c("numLayers", "sizeHidden", "hiddenFactor",
                      "residualDropout", "hiddenDropout", "sizeEmbedding")
  modelParam <- param[modelParamNames]

  fitParamNames <- c("weightDecay", "learningRate", "epochs", "batchSize")
  fitParams <- param[fitParamNames]
  
  n_features <- ncol(sparseMatrix$data)
  modelParam$n_features <- n_features
  
  # TODO make more general for other variables than only age
  numericalIndex <- sparseMatrix$map$newCovariateId[sparseMatrix$map$oldCovariateId==1002]
  
  index_vect <- unique(population$indexes[population$indexes > 0])
  if(train==T){
    ParallelLogger::logInfo(paste('Training deep neural network using Torch with ',length(index_vect),' fold CV'))
    foldAuc <- c()
    foldEpochs <- c()
    for(index in 1:length(index_vect)){
      ParallelLogger::logInfo(paste('Fold ',index, ' -- with ', sum(population$indexes!=index & population$indexes > 0),'train rows'))
      testIndices <- population$rowId[population$indexes==index]
      trainIndices <- population$rowId[(population$indexes!=index) & (population$indexes > 0)]
      trainDataset <- Dataset(sparseMatrix$data[population$rowId,], 
                              population$outcomeCount, 
                              indices= population$rowId%in%trainIndices, 
                              numericalIndex=numericalIndex)
      testDataset <- Dataset(sparseMatrix$data[population$rowId,], 
                             population$outcomeCount, 
                             indices = population$rowId%in%testIndices, 
                             numericalIndex = numericalIndex)
      fitParams['posWeight'] <- trainDataset$posWeight
      estimator <- Estimator$new(baseModel=ResNet, 
                                 modelParameters=modelParam,
                                 fitParameters=fitParams, 
                                 device=param$device)
      estimator$fit(trainDataset, testDataset)
      score <- estimator$bestScore
      bestEpoch <- estimator$bestEpoch
      auc <- score$auc
      foldAuc <- c(foldAuc, auc)
      foldEpochs <- c(foldEpochs, bestEpoch)
    }
    auc <- mean(foldAuc)
    prediction <- NULL
    bestEpochs <- list(bestEpochs=foldEpochs)
  }
  else {
    ParallelLogger::logInfo('Training deep neural network using Torch on whole training set')
    fitParams$resultsDir <- param$resultsDir    
    estimator <- Estimator$new(baseModel = ResNet,
                               modelParameters = modelParam,
                               fitParameters = fitParams, 
                               device=param$device)
    
    trainIndices <- population$rowId[population$indexes > 0]
    
    trainDataset <- Dataset(sparseMatrix$data[population$rowId,], 
                            population$outcomeCount, 
                            indices=population$rowId%in%trainIndices, 
                            numericalIndex=numericalIndex)
   
    estimator$fitWholeTrainingSet(trainDataset)
    
    # get predictions
    prediction <- population[population$rowId%in%trainIndices, ]
    prediction$value <- estimator$predictProba(trainDataset)
    
    #predictionsClass <- data.frame(value=predictions$value, 
    #                               outcomeCount=as.array(trainDataset$labels))
    
    attr(prediction, 'metaData')$predictionType <-'binary' 
    auc <- computeAuc(prediction)
    bestEpochs <- NULL
  }
 
   result <- list(model = estimator,
                 auc = auc,
                 prediction = prediction,
                 hyperSum = c(modelParam, fitParams, bestEpochs))

  return(result)
  }

ResLayer <- torch::nn_module(
  name='ResLayer',
  
  initialize=function(sizeHidden, resHidden, normalization,
                     activation, hiddenDropout=NULL, residualDropout=NULL){
    self$norm <- normalization(sizeHidden)
    self$linear0 <- torch::nn_linear(sizeHidden, resHidden)
    self$linear1 <- torch::nn_linear(resHidden, sizeHidden)
    
    self$activation <- activation
    if (!is.null(hiddenDropout)){
      self$hiddenDropout <- torch::nn_dropout(p=hiddenDropout)
    }
    if (!is.null(residualDropout)) 
    {
      self$residualDropout <- torch::nn_dropout(p=residualDropout)
    }
    
    self$activation <- activation()
    
  },
  
  forward=function(x) {
    z <- x
    z <- self$norm(z)
    z <- self$linear0(z)
    z <- self$activation(z)
    if (!is.null(self$hiddenDropout)) {
      z <- self$hiddenDropout(z)
    }
    z <- self$linear1(z)
    if (!is.null(self$residualDropout)) {
      z <- self$residualDropout(z)
    }
    x <- z + x 
    return(x)
  }
)

ResNet <- torch::nn_module(
  name='ResNet',
  
  initialize=function(n_features, sizeEmbedding, sizeHidden, numLayers,
                      hiddenFactor, activation=torch::nn_relu, 
                      normalization=torch::nn_batch_norm1d, hiddenDropout=NULL,
                      residualDropout=NULL, d_out=1) {
    # n_features - 1 because only binary features are embedded (not Age)
    # ages is concatenated with the embedding output
    # TODO need to extend to support other numerical features
    self$embedding <- torch::nn_linear(n_features - 1, sizeEmbedding, bias=F)
    self$first_layer <- torch::nn_linear(sizeEmbedding + 1, sizeHidden)
    
    resHidden <- sizeHidden * hiddenFactor
    
    self$layers <- torch::nn_module_list(lapply(1:numLayers,
                                                 function (x) ResLayer(sizeHidden, resHidden,
                                                          normalization, activation,
                                                          hiddenDropout,
                                                          residualDropout)))
    self$lastNorm <- normalization(sizeHidden)
    self$head <- torch::nn_linear(sizeHidden, d_out)
    
    self$lastAct <- activation()
    
  },
      
  forward=function(x_num, x_cat) {
    x_cat <- self$embedding(x_cat)
    x <- torch::torch_cat(list(x_cat, x_num), dim=2L)
    x <- self$first_layer(x)
    
    for (i in 1:length(self$layers)) {
      x <- self$layers[[i]](x)
    }
    x <- self$lastNorm(x)
    x <- self$lastAct(x)    
    x <- self$head(x)
    x <- x$squeeze(-1)
    return(x)
  }
)

  
  
  
