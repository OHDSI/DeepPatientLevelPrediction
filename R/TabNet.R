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
    modelType = 'DeepNNTorch',
    seed = seed,
    name = "TabNetTorch",
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
  if (!FeatureExtraction::isCovariateData(plpData$covariateData)){
    stop('TabNetTorch requires correct covariateData')
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
    what = gridCvTabNetTorch,
    args = list(
      matrixData = matrixData,
      labels = labels,
      seed = settings$seed,
      modelName = settings$name,
      device = settings$device,
      batch_size = settings$batch_size,
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

gridCvTabNetTorch <- function(
  matrixData,
  labels,
  seed,
  batch_size,
  epochs,
  drop_last,
  clip_value,
  loss,
  mask_type,
  optimizer,
  lr_scheduler,
  verbose,
  device,
  paramSearch
){
  
  
  ParallelLogger::logInfo(paste0("Running CV for ",modelName," model"))
  
  ###########################################################################
  
  
  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- length(paramSearch)
  
  for(gridId in 1:length(paramSearch)){
    
    # get the params
    
    config <- tabnet_config(batch_size = batch_size,
                            penalty = paramSearch$penalty,
                            clip_value = clip_value,
                            loss = loss,
                            epochs = epochs,
                            drop_last = drop_last,
                            decision_width = paramSearch$decision_width,
                            attention_width = paramSearch$attention_width,
                            num_steps = paramSearch$num_steps,
                            feature_reusage = paramSearch$feature_reusage,
                            mask_type = mask_type,
                            virtual_batch_size = paramSearch$virtual_batch_size,
                            valid_split = paramSearch$valid_split,
                            learn_rate = paramSearch$learn_rate,
                            optimizer = optimizer,
                            lr_scheduler = lr_scheduler,
                            lr_decay = paramSearch$lr_decay,
                            step_size = paramSearch$step_size,
                            checkpoint_epochs = paramSearch$checkpoint_epochs,
                            cat_emb_dim = paramSearch$cat_emb_dim,
                            num_independent = paramSearch$num_independent,
                            num_shared = paramSearch$num_shared,
                            momentum = paramSearch$momentum,
                            pretraining_ratio = paramSearch$pretraining_ratio,
                            verbose = verbose,
                            device = device,
                            importance_sample_size = paramSearch$importance_sample_size,
                            seed = seed) 
    
    
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
      testLabel <- labels[fold == i,]
      
      model <- tabnet_fit(x = trainDataset, y = trainLabel$outcomeCount, config = config)
      
      ParallelLogger::logInfo("Calculating predictions on left out fold set...")
      
      pred <- predict(model, testDataset)
      predictionTable <- testLabel
      predictionTable$value <- pred$.pred
      
      if(!'plpModel' %in% class(model)){
        model <- list(model = model)
        attr(model, 'modelType') <- 'binary'
      }
      attr(predictionTable, "metaData")$modelType <-  attr(model, 'modelType')
      
      prediction <- rbind(prediction, predictionTable)
      
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
  
  finalParam$batch_size = batch_size
  finalParam$epochs = epochs
  finalParam$drop_last = drop_last
  finalParam$clip_value = clip_value
  finalParam$loss = loss
  finalParam$mask_type = mask_type
  finalParam$ optimizer = optimizer
  finalParam$lr_scheduler = lr_scheduler
  finalParam$verbose = verbose
  finalParam$device = device
  
  config <- tabnet_config(finalParam) 
  
  # create the dir
  if(!dir.exists(file.path(modelLocation))){
    dir.create(file.path(modelLocation), recursive = T)
  }
  
  trainDataset <- as.data.frame(as.matrix(matrixData))
  trainLabel <- labels

  model <- tabnet_fit(x = trainDataset, y = trainLabel$outcomeCount, config = config)
  
  ParallelLogger::logInfo("Calculating predictions on all train data...")
  
  pred <- predict(model, trainDataset)
  prediction <- trainLabel
  predictionTable$value <- pred$.pred
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
  torch_save(model, file.path(modelLocation, 'TabNetTorchModel.rds'))
  
  return(
    list( 
      estimator = modelLocation,
      prediction = prediction,
      finalParam = finalParam,
      paramGridSearch = paramGridSearch
    )
  )
  
}

#' @export
predictTabNetTorch <- function(
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
    
    data <- as.data.frame(as.matrix(dataMat$dataMatrix))
  }
  
  # get predictions
  prediction <- cohort
  
  if(is.character(plpModel$model)) model <- torch::torch_load(file.path(plpModel$model, 'TabNetTorchModel.rds'), device='cpu')
  
  pred <- predict(model, data)
  prediction$value <- pred$.pred
  
  attr(prediction, "metaData")$modelType <- attr(plpModel, 'modelType')
  
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


