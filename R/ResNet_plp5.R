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

#' setResNet_plp5
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
setResNet_plp5 <- function(
  numLayers = list(1:16), 
  sizeHidden = list(2^(6:10)), 
  hiddenFactor = list(1:4),
  residualDropout = list(seq(0,0.3,0.05)), 
  hiddenDropout = list(seq(0,0.3,0.05)),
  normalization = list('BatchNorm'), 
  activation = list('RelU'),
  sizeEmbedding = list(2^(6:9)), 
  weightDecay = list(c(1e-6, 1e-3)),
  learningRate = list(c(1e-2,1e-5)), 
  seed = NULL, 
  hyperParamSearch = 'random',
  randomSample = 100, 
  device = 'cpu', 
  batchSize = 1024, 
  epochs = 10
  ) {

  if (!is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  
  paramGrid <- list(
    numLayers = numLayers, 
    sizeHidden = sizeHidden,
    hiddenFactor = hiddenFactor,
    residualDropout = residualDropout,
    hiddenDropout = hiddenDropout,
    sizeEmbedding = sizeEmbedding, 
    weightDecay = weightDecay,
    learningRate = learningRate,
    seed = list(as.integer(seed[[1]]))
  )
  
  param <- listCartesian(paramGrid)
  
  if (hyperParamSearch=='random'){
    param <- param[sample(length(param), randomSample)]
  }

  attr(param, 'settings') <- list(
    seed = seed[1],
    device = device,
    batchSize = batchSize,
    epochs = epochs,
    name = "ResNet",
    saveType = 'file'
  )

  results <- list(
    fitFunction = 'fitResNet_plp5',
    param = param
  )

  class(results) <- 'modelSettings'

  return(results)

}

#' fitResNet_plp5
#'
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
fitResNet_plp5 <- function(
  trainData, 
  param, 
  search = 'grid',
  analysisId,
  ...
) {
  
  start <- Sys.time()
  
  # check covariate data
  if(!FeatureExtraction::isCovariateData(trainData$covariateData)){stop("Needs correct covariateData")}
  
  # get the settings from the param
  settings <- attr(param, 'settings')
  
  if(!is.null(trainData$folds)){
    trainData$labels <- merge(trainData$labels, trainData$fold, by = 'rowId')
  }
  
  mappedData <- PatientLevelPrediction::toSparseM(
    plpData = trainData,  
    map = NULL
    )
  
  matrixData <- mappedData$dataMatrix
  labels <- mappedData$labels
  covariateRef <- mappedData$covariateRef
  
  outLoc <- PatientLevelPrediction:::createTempModelLoc() # export
  
  cvResult <- do.call( 
    what = gridCvDeep,
    args = list(
      matrixData = matrixData,
      labels = labels,
      seed = settings$seed,
      modelName = settings$name,
      device = settings$device,
      batchSize = settings$batchSize,
      epochs = settings$epochs,
      modelLocation = outLoc,
      paramSearch = param
    )
  )
  
  hyperSummary <- do.call(rbind, lapply(cvResult$paramGridSearch, function(x) x$hyperSummary))
  
  prediction <- cvResult$prediction
  
  incs <- rep(1, nrow(covariateRef))
  covariateRef$included <- incs
  covariateRef$covariateValue <- 0
  
  comp <- start - Sys.time()
  
  result <- list(
    model = cvResult$estimator, #file.path(outLoc),
    
    prediction = prediction,
    
    settings = list(
      plpDataSettings = attr(trainData, "metaData")$plpDataSettings,
      covariateSettings = attr(trainData, "metaData")$covariateSettings,
      populationSettings = attr(trainData, "metaData")$populationSettings,
      featureEngineering = attr(trainData$covariateData, "metaData")$featureEngineering,
      tidyCovariates = attr(trainData$covariateData, "metaData")$tidyCovariateDataSettings, 
      requireDenseMatrix = F,
      modelSettings = list(
        model = settings$name, 
        param = param,
        finalModelParameters = cvResult$finalParam,
        extraSettings = attr(param, 'settings')
      ),
      splitSettings = attr(trainData, "metaData")$splitSettings,
      sampleSettings = attr(trainData, "metaData")$sampleSettings
    ),
    
    trainDetails = list(
      analysisId = analysisId,
      cdmDatabaseSchema = attr(trainData, "metaData")$cdmDatabaseSchema,
      outcomeId = attr(trainData, "metaData")$outcomeId,
      cohortId = attr(trainData, "metaData")$cohortId,
      attrition = attr(trainData, "metaData")$attrition, 
      trainingTime = comp,
      trainingDate = Sys.Date(),
      hyperParamSearch = hyperSummary
    ),
    
    covariateImportance = covariateRef
  )
  
  class(result) <- "plpModel"
  attr(result, "predictionFunction") <- "predictDeepEstimator"
  attr(result, "modelType") <- "binary"
  attr(result, "saveType") <- attr(param, 'saveType')
  
  return(result)
}

#' predictDeepEstimator
#'
#' @description 
#' the prediction function for the binary classification deep learning models
#' 
#' @param plpModel    the plpModel
#' @param data       plp data object or a torch dataset
#' @param cohort     a data.frame with the rowIds of the people to predict risk for
#'
#' @export 
predictDeepEstimator <- function(
  plpModel, 
  data, 
  cohort
){
  
  if(!'plpModel' %in% class(plpModel)){
    plpModel <- list(model = plpModel)
    attr(plpModel, 'modelType') <- 'binary'
  }
  
  if("plpData" %in% class(data)){
    
    dataMat <- PatientLevelPrediction::toSparseM(
      plpData = data, 
      cohort = cohort, 
      map = plpModel$covariateImportance %>% 
        dplyr::select(.data$columnId, .data$covariateId)
        )
    
    data <- Dataset_plp5(dataMat$dataMatrix) # add numeric details..
  }
  
  # get predictions
  prediction <- cohort
  prediction$value <- plpModel$model$predictProba(data)
  
  attr(prediction, "metaData")$modelType <-  attr(plpModel, 'modelType')
  
  return(prediction)
}


gridCvDeep <- function(
  matrixData,
  labels,
  seed,
  modelName,
  device,
  batchSize,
  epochs,
  modelLocation,
  paramSearch
){
  
  
  ParallelLogger::logInfo(paste0("Rnning CV for ",modelName," model"))
  
  ###########################################################################
  
  
  n_features <- ncol(matrixData)

  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- length(paramSearch)
  
  for(gridId in 1:length(paramSearch)){
    
    # get the params
    modelParamNames <- c("numLayers", "sizeHidden", "hiddenFactor",
      "residualDropout", "hiddenDropout", "sizeEmbedding")
    modelParams <- paramSearch[[gridId]][modelParamNames]
    modelParams$n_features <- n_features
    
    fitParams <- paramSearch[[gridId]][c("weightDecay", "learningRate")]
    fitParams$epochs <- epochs
    fitParams$batchSize <- batchSize
    
    
    # initiate prediction
    prediction <- c()
    
    fold <- labels$index
    ParallelLogger::logInfo(paste0('Max fold: ', max(fold)))
    
    for( i in 1:max(fold)){
      
      ParallelLogger::logInfo(paste0('Fold ',i))
      trainDataset <- Dataset_plp5(
        matrixData[fold != i,],
        labels$outcomeCount[fold != i]
        )
      testDataset <- Dataset_plp5(
        matrixData[fold == i,],
        labels$outcomeCount[fold == i], 
        trainDataset$getNumericalIndex
      )
      
      fitParams['posWeight'] <- trainDataset$posWeight
      
      estimator <- Estimator$new(
        baseModel = ResNet, 
        modelParameters = modelParams,
        fitParameters = fitParams, 
        device = device
        )
      
      estimator$fit(
        trainDataset, 
        testDataset
        )
      
      ParallelLogger::logInfo("Calculating predictions on left out fold set...")
      
      prediction <- rbind(
        prediction, 
        predictDeepEstimator(
          plpModel = estimator, 
          data = testDataset, 
          cohort = labels[fold == i,]
          )
        )
      
    }
    
    gridSearchPredictons[[gridId]] <- list(
      prediction = prediction,
      param = paramSearch[[gridId]]
    )
  }
  
  # get best para (this could be modified to enable any metric instead of AUC, just need metric input in function)
  
  paramGridSearch <- lapply(gridSearchPredictons, function(x){do.call(computeGridPerformance, x)})  # cvAUCmean, cvAUC, param
  
  optimalParamInd <- which.max(unlist(lapply(paramGridSearch, function(x) x$cvPerformance)))
  
  finalParam <- paramGridSearch[[optimalParamInd]]$param
  
  cvPrediction <- gridSearchPredictons[[optimalParamInd]]$prediction
  cvPrediction$evaluationType <- 'CV'
  
  ParallelLogger::logInfo('Training final model using optimal parameters')
  
  # get the params
  modelParamNames <- c("numLayers", "sizeHidden", "hiddenFactor",
    "residualDropout", "hiddenDropout", "sizeEmbedding")
  modelParams <- finalParam[modelParamNames]
  modelParams$n_features <- n_features
  fitParams <- finalParam[c("weightDecay", "learningRate")]
  fitParams$epochs <- epochs
  fitParams$batchSize <- batchSize
  fitParams$resultsDir <- modelLocation # remove this?
  # create the dir
  if(!dir.exists(file.path(modelLocation))){
    dir.create(file.path(modelLocation), recursive = T)
  }
  
  estimator <- Estimator$new(
    baseModel = ResNet,
    modelParameters = modelParams,
    fitParameters = fitParams, 
    device = device
    )
  
  trainDataset <- Dataset_plp5(
    matrixData, 
    labels$outcomeCount
    )
  
  numericalIndex <- trainDataset$getNumericalIndex
  
  estimator$fitWholeTrainingSet(trainDataset)
  
  ParallelLogger::logInfo("Calculating predictions on all train data...")
  prediction <- predictDeepEstimator(
    plpModel = estimator, 
    data = trainDataset, 
    cohort = labels
  )
  prediction$evaluationType <- 'Train'
  
  prediction <- rbind(
    prediction,
    cvPrediction
  )
  
  # modify prediction 
  prediction <- prediction %>% 
    dplyr::select(-.data$rowId, -.data$index) %>%
    dplyr::rename(rowId = .data$originalRowId)
  
  prediction$cohortStartDate <- as.Date(prediction$cohortStartDate, origin = '1970-01-01')
  

  # save torch code here
  
  
  return(
    list( 
      estimator = estimator,
      prediction = prediction,
      finalParam = finalParam,
      paramGridSearch = paramGridSearch,
      numericalIndex = numericalIndex
    )
  )
  
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

  



listCartesian <- function(allList){
  
  sizes <- lapply(allList, function(x) 1:length(x))
  combinations <- expand.grid(sizes)
  
  result <- list()
  length(result) <- nrow(combinations)
  
  for(i in 1:nrow(combinations)){
    tempList <- list()
    for(j in 1:ncol(combinations)){
      tempList <- c(tempList, list(allList[[j]][[combinations[i,j]]]))
    }
    names(tempList) <- names(allList)
    result[[i]] <- tempList
  }
  
  return(result)
}


# export this in PLP
computeGridPerformance <- PatientLevelPrediction:::computeGridPerformance
  
  
