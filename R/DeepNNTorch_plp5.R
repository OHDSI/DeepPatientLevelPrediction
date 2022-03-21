#' @export
setDeepNNTorch <- function(
  units=list(c(128, 64), 128),
  layer_dropout=c(0.2),
  lr =c(1e-4),
  decay=c(1e-5),
  outcome_weight = c(1.0),
  batch_size = c(10000), 
  epochs= c(100),
  device = 'cpu', 
  seed=NULL){
  
  # ensure_installed("torch")
  
  param <- expand.grid(units=units,
                       layer_dropout=layer_dropout,
                       lr =lr, decay=decay, outcome_weight=outcome_weight, epochs= epochs,
                       seed=ifelse(is.null(seed),'NULL', seed))
  
  param$units1=unlist(lapply(param$units, function(x) x[1])) 
  param$units2=unlist(lapply(param$units, function(x) x[2])) 
  param$units3=unlist(lapply(param$units, function(x) x[3]))
  
  attr(param, 'settings') <- list(
    selectorType = "byPid",  # is this correct?
    crossValidationInPrior = T,
    modelType = 'DeepNN',
    seed = seed[1],
    name = "DeepNNTorch",
    units = units,
    layer_dropout = layer_dropout,
    lr = lr,
    decay = decay,
    outcome_weight = outcome_weight,
    batch_size = batch_size,
    device = device,
    epochs = epochs
  )
  
  attr(param, 'modelType') <- 'binary' 
  attr(param, 'saveType') <- 'RtoJson'
  
  result <- list(fitFunction='fitDeepNNTorch', 
                 param=param)
  
  class(result) <- 'modelSettings' 
  
  return(result)
  
}

#' @export
fitDeepNNTorch <- function(
  trainData,
  param, 
  search='grid', 
  analysisId,
  ...){
  
  start <- Sys.time()
  
  # check covariateData
  if (!FeatureExtraction::isCovariateData(plpData$covariateData)){
    stop('DeepNNTorch requires correct covariateData')
  }
  
  # get the settings from the param
  settings <- attr(param, 'settings')
  
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
    what = gridCvDeepNN,
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
  attr(result, "predictionFunction") <- "predictDeepNN"
  attr(result, "modelType") <- "binary"
  attr(result, "saveType") <- attr(param, 'settings')$saveType
  
  return(result)
}

#' @export
predictDeepNN <- function(
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
  
  if(is.character(plpModel$model)) model <- torch::torch_load(file.path(plpModel$model, 'DeepNNTorchModel.rds'), device='cpu')
    
  y_pred = model(data$all)
  prediction$value <- as.array(y_pred$to())[,1]
    
  attr(prediction, "metaData")$modelType <- attr(plpModel, 'modelType')
  
  return(prediction)
}


gridCvDeepNN <- function(
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
  
  
  ParallelLogger::logInfo(paste0("Running CV for ",modelName," model"))
  
  ###########################################################################
  
  
  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- nrow(paramSearch)
  
  for(gridId in 1:nrow(paramSearch)){
    
    # get the params
    modelParamNames <- c("layer_dropout", "lr", "decay", "outcome_weight", "epochs", "units1", "units2", "units3")
    modelParams <- paramSearch[gridId, modelParamNames]
    
    fitParams <- paramSearch[gridId, c("lr", "decay")]
    fitParams$epochs <- epochs
    fitParams$batchSize <- batchSize
    
    
    # initiate prediction
    prediction <- c()
    
    fold <- labels$index
    ParallelLogger::logInfo(paste0('Max fold: ', max(fold)))
    
    dataset <- Dataset_plp5(matrixData, labels$outcomeCount)
    # modelParams$cat_features <- dataset$cat$shape[2]
    # modelParams$num_features <- dataset$num$shape[2]
    
     for(i in 1:max(fold)){
     
       if(is.na(modelParams$units2)){
         model <- singleLayerNN(inputN = ncol(matrixData),
                                layer1 = modelParams$units1, 
                                outputN = 2, 
                                layer_dropout = modelParams$layer_dropout)
         
       } else if(is.na(modelParams$units3)){
         model <- doubleLayerNN(inputN = ncol(matrixData),
                                layer1 = modelParams$units1,
                                layer2 = modelParams$units2,
                                outputN = 2, 
                                layer_dropout = modelParams$layer_dropout)
       } else{
         model <- tripleLayerNN(inputN = ncol(matrixData),
                                layer1 = modelParams$units1,
                                layer2 = modelParams$units2,
                                layer3 = modelParams$units3,
                                outputN = 2, 
                                layer_dropout = modelParams$layer_dropout)
       }
       
       criterion = torch::nn_bce_loss() #Binary crossentropy only
       optimizer = torch::optim_adam(model$parameters, lr = fitParams$lr)
       
       # Need earlyStopping
       # Need setting decay
       
      ParallelLogger::logInfo(paste0('Fold ',i))
      trainDataset <- torch::dataset_subset(dataset, indices=which(fold!=i)) 
      testDataset <- torch::dataset_subset(dataset, indices=which(fold==i))
      
      # batches <- split(trainDataset, ceiling(seq_along(trainDataset)/batch_size))
      
      for(j in 1:epochs){
        # for(batchRowIds in batches){
          optimizer$zero_grad()
          y_pred = model(trainDataset$dataset$all[trainDataset$indices])
          loss = criterion(y_pred[,1], trainDataset$dataset$target[trainDataset$indices])
          loss$backward()
          optimizer$step()
          
          if(j%%1 == 0){
            cat("Epoch:", j, "out of ", epochs , ": Loss:", loss$item(), "\n")
          }
        # }
      }
          model$eval()
          
          ParallelLogger::logInfo("Calculating predictions on left out fold set...")
          
          pred <- model(testDataset$dataset$all[testDataset$indices])
          predictionTable <- labels[labels$index == i,]
          predictionTable$value <- as.array(pred$to())[,1]
          
          if(!'plpModel' %in% class(model)){
            model <- list(model = model)
            attr(model, 'modelType') <- 'binary'
          }
          attr(predictionTable, "metaData")$modelType <-  attr(model, 'modelType')
          
          prediction <- rbind(prediction, predictionTable)

  }
    gridSearchPredictons[[gridId]] <- list(
      prediction = prediction,
      param = paramSearch[gridId,]
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
  modelParamNames <- c("layer_dropout", "lr", "decay", "outcome_weight", "epochs", "units1", "units2", "units3")
  modelParams <- finalParam[modelParamNames]
  fitParams <- finalParam[c("lr", "decay")]
  fitParams$epochs <- epochs
  fitParams$batchSize <- batchSize
  # create the dir
  if(!dir.exists(file.path(modelLocation))){
    dir.create(file.path(modelLocation), recursive = T)
  }
  
  trainDataset <- Dataset_plp5(
    matrixData, 
    labels$outcomeCount
  )
  
  # modelParams$cat_features <- trainDataset$cat$shape[2]
  # modelParams$num_features <- trainDataset$num$shape[2]
  
  # trainDataset <- torch::dataset_subset(dataset, indices=which(fold!=i)) 
  
  if(is.na(modelParams$units2)){
    model <- singleLayerNN(inputN = ncol(matrixData),
                           layer1 = modelParams$units1, 
                           outputN = 2, 
                           layer_dropout = modelParams$layer_dropout)
    
  } else if(is.na(modelParams$units3)){
    model <- doubleLayerNN(inputN = ncol(matrixData),
                           layer1 = modelParams$units1,
                           layer2 = modelParams$units2,
                           outputN = 2, 
                           layer_dropout = modelParams$layer_dropout)
  } else{
    model <- tripleLayerNN(inputN = ncol(matrixData),
                           layer1 = modelParams$units1,
                           layer2 = modelParams$units2,
                           layer3 = modelParams$units3,
                           outputN = 2, 
                           layer_dropout = modelParams$layer_dropout)
  }
  
  criterion = torch::nn_bce_loss() #Binary crossentropy only
  optimizer = torch::optim_adam(model$parameters, lr = fitParams$lr)
  optimizer$zero_grad()
  y_pred = model(trainDataset$all)
  loss = criterion(y_pred[,1], trainDataset$target)
  loss$backward()
  optimizer$step()
  model$eval()
  
  ParallelLogger::logInfo("Calculating predictions on all train data...")
 
  prediction <- labels
  prediction$value <- as.array(y_pred$to())[,1]
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
  torch_save(model, file.path(modelLocation, 'DeepNNTorchModel.rds'))
  
  return(
    list( 
      estimator = modelLocation,
      prediction = prediction,
      finalParam = finalParam,
      paramGridSearch = paramGridSearch
    )
  )
  
  }
  
  
