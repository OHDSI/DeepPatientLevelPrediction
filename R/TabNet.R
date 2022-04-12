# @file TabNet.R
#
# Copyright 2020 Observational Health Data Sciences and Informatics
#
# This file is part of DeepPatientLevelPrediction
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

#' @export
setTabNetTorch <- function(
  batch_size = 256,
  penalty = 1e-3,
  clip_value = NULL,
  loss = "auto",
  epochs = 5,
  drop_last = FALSE,
  decision_width = 8,
  attention_width = 8,
  num_steps = 3,
  feature_reusage = 1.3,
  mask_type = "sparsemax",
  virtual_batch_size = 128,
  valid_split = 0,
  learn_rate = 2e-2,
  optimizer = "adam",
  lr_scheduler = NULL,
  lr_decay = 0.1,
  step_size = 30,
  checkpoint_epochs = 10,
  cat_emb_dim = 1,
  num_independent = 2,
  num_shared = 2,
  momentum = 0.02,
  pretraining_ratio = 0.5,
  verbose = FALSE,
  device = "auto",
  importance_sample_size = 1e5,
  seed=NULL,
  hyperParamSearch = 'random',
  randomSample = 100){
  
  # ensure_installed("torch")
  
  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  
  paramGrid <- list(
    penalty = penalty,
    decision_width = decision_width,
    attention_width = attention_width,
    num_steps = num_steps,
    feature_reusage = feature_reusage,
    virtual_batch_size = virtual_batch_size,
    valid_split = valid_split,
    learn_rate = learn_rate,
    lr_decay = lr_decay,
    step_size = step_size,
    checkpoint_epochs = checkpoint_epochs,
    cat_emb_dim = cat_emb_dim,
    num_independent = num_independent,
    num_shared = num_shared,
    momentum = momentum,
    pretraining_ratio = pretraining_ratio,
    importance_sample_size = importance_sample_size
  )
  
  param <- listCartesian(paramGrid)

  # if (hyperParamSearch=='random'){
  #   param <- param[sample(length(param), randomSample)]
  # }
  
  attr(param, 'settings') <- list(
    batch_size = batch_size,
    epochs = epochs,
    drop_last = drop_last,
    clip_value = clip_value,
    loss = loss,
    mask_type = mask_type,
    optimizer = optimizer,
    lr_scheduler = lr_scheduler,
    verbose = verbose,
    device = device
  )
  
  attr(param, 'modelType') <- 'binary' 
  attr(param, 'saveType') <- 'file'
  attr(param, 'name') <-  "TabNetTorch"
  
  result <- list(fitFunction='fitTabNetTorch', 
                 param=param)
  
  class(result) <- 'modelSettings' 
  
  return(result)
}

#' @export
fitTabNetTorch <- function(
  trainData,
  param, 
  search='grid', 
  analysisId,
  ...){
  
  start <- Sys.time()
  
  # check covariateData
  if (!FeatureExtraction::isCovariateData(trainData$covariateData)){
    stop('TabNetTorch requires correct covariateData')
  }
  
  
  if(!is.null(trainData$folds)){
    trainData$labels <- merge(trainData$labels, trainData$folds, by = 'rowId')
  }
  
  mappedData <- PatientLevelPrediction::toSparseM(
    plpData = trainData,
    map = NULL
  )
  
  matrixData <- mappedData$dataMatrix
  labels <- mappedData$labels
  covariateRef <- mappedData$covariateRef
  
  outLoc <- PatientLevelPrediction:::createTempModelLoc() #export
  
  cvResult <- do.call( 
    what = gridCvTabNetTorch,
    args = list(
      matrixData = matrixData,
      labels = labels,
      modelLocation = outLoc,
      paramSearch = param
    )
  )
  hyperSummary <- do.call(rbind, lapply(cvResult$paramGridSearch, function(x) x$hyperSummary))
  
  prediction <- cvResult$prediction
  
  variableImportance <- cvResult$variableImportance
  incs <- seq_len(nrow(variableImportance))
  variableImportance$columnId <- incs

  browser()
  covariateRef <- covariateRef %>% merge(variableImportance, by = 'columnId', 
                                         all.x = TRUE)  %>%
                                   dplyr::mutate(included=1)  %>%
                                   dplyr::rename(covariateValue=importance) %>%
                                   dplyr::select(!variables)
  covariateRef$covariateValue[is.na(covariateRef$covariateValue)] <- 0
  covariateRef$included[is.na(covariateRef$included)] <- 0
  
  
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
        model = attr(param, 'name'), 
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
  attr(result, "predictionFunction") <- "predictTabNetTorch"
  attr(result, "modelType") <- "binary"
  attr(result, "saveType") <- attr(param, 'saveType')
  
  return(result)
}

gridCvTabNetTorch <- function(
  matrixData,
  labels,
  paramSearch,
  modelLocation
){
  
  fitSettings <- attr(paramSearch, 'settings')
  
  ParallelLogger::logInfo(paste0("Running CV for ",attr(paramSearch, 'name')," model"))
  
  ###########################################################################
  
  
  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- length(paramSearch)
  
  for(gridId in 1:length(paramSearch)){

    # get the params
    config <- do.call(tabnet::tabnet_config, args=c(paramSearch[[gridId]], fitSettings))
    
    # initiate prediction
    prediction <- c()
    
    fold <- labels$index
    ParallelLogger::logInfo(paste0('Max fold: ', max(fold)))
    
    # dataset <- Dataset_plp5(matrixData, labels$outcomeCount)
    # modelParams$cat_features <- dataset$cat$shape[2]
    # modelParams$num_features <- dataset$num$shape[2]
    
    # rec <- recipes::recipe(dataset$target ~ ., data = dataset$all)
    # fit <- tabnet_fit(x = as.data.frame(as.matrix(matrixData)), y =labels$outcomeCount , epoch = epochs)
    
    for(i in 1:max(fold)){
      
      ParallelLogger::logInfo(paste0('Fold ',i))
      trainDataset <- as.data.frame(as.matrix(matrixData)[fold != i,])
      trainLabel <- labels[fold != i,]
      
      testDataset <-as.data.frame(as.matrix(matrixData)[fold == i,])

      model <- tabnet::tabnet_fit(x = trainDataset, y = trainLabel$outcomeCount, config = config)
      
      ParallelLogger::logInfo("Calculating predictions on left out fold set...")
      
      prediction <- rbind(prediction, predictTabNetTorch(plpModel = model,
                                                         data = testDataset,
                                                         cohort = labels[fold == i,]))
      
    }
    
    gridSearchPredictons[[gridId]] <- list(
      prediction = prediction,
      param = paramSearch[[gridId]]
    )
  }    
  
  
  # get best para (this could be modified to enable any metric instead of AUC, just need metric input in function)
  
  paramGridSearch <- lapply(gridSearchPredictons, function(x){do.call(PatientLevelPrediction:::computeGridPerformance, x)})  # cvAUCmean, cvAUC, param
  
  optimalParamInd <- which.max(unlist(lapply(paramGridSearch, function(x) x$cvPerformance)))
  
  finalParam <- paramGridSearch[[optimalParamInd]]$param
  
  cvPrediction <- gridSearchPredictons[[optimalParamInd]]$prediction
  cvPrediction$evaluationType <- 'CV'
  
  ParallelLogger::logInfo('Training final model using optimal parameters')
  # get the params
  finalParam <- c(finalParam, fitSettings)
  
  config <- do.call(tabnet::tabnet_config, finalParam) 
  
  # create the dir
  if(!dir.exists(file.path(modelLocation))){
    dir.create(file.path(modelLocation), recursive = T)
  }
  
  trainDataset <- as.data.frame(as.matrix(matrixData))

  model <- tabnet::tabnet_fit(x = trainDataset, y = labels$outcomeCount, config = config)
  
  ParallelLogger::logInfo("Calculating predictions on all train data...")
  
  prediction <- predictTabNetTorch(plpModel = model, 
                                   data = trainDataset,
                                   cohort = labels)
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
  saveRDS(model, file.path(modelLocation, 'TabNetTorchModel.Rds'))
  return(
    list( 
      estimator = modelLocation,
      prediction = prediction,
      finalParam = finalParam,
      paramGridSearch = paramGridSearch,
      variableImportance = model$fit$importances
    )
  )
  
}

#' @export
predictTabNetTorch <- function(
  plpModel,
  data,
  cohort
){

  if("plpData" %in% class(data)){
    
    dataMat <- PatientLevelPrediction::toSparseM(
      plpData = data, 
      cohort = cohort, 
      map = plpModel$covariateImportance %>% 
        dplyr::select(.data$columnId, .data$covariateId)
    )
    
    data <- as.data.frame(as.matrix(dataMat$dataMatrix))
  }
  
  # get predictions
  prediction <- cohort
  if(is.character(plpModel$model)) {
    plpModel <- readRDS(file.path(plpModel$model, 'TabNetTorchModel.Rds'))
  }
    
  
  pred <- predict(plpModel, data)
  prediction$value <- as.vector(as.matrix(torch::torch_sigmoid(pred$.pred)))
  
  attr(prediction, "metaData")$modelType <- 'binary'
  
  return(prediction)
}

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


