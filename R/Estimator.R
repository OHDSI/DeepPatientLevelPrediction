# @file Estimator.R
#
# Copyright 2022 Observational Health Data Sciences and Informatics
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

#' setEstimator
#'
#' @description
#' creates settings for the Estimator, which takes a model and trains it
#'
#' @name setEstimator
#' @param learningRate  what learning rate to use
#' @param weightDecay what weight_decay to use
#' @param batchSize batchSize to use
#' @param epochs  how many epochs to train for
#' @param device  what device to train on, can be a string or a function to that evaluates
#' to the device during runtime
#' @param optimizer which optimizer to use
#' @param scheduler which learning rate scheduler to use
#' @param criterion loss function to use
#' @param earlyStopping If earlyStopping should be used which stops the training of your metric is not improving
#' @param metric either `auc` or `loss` or a custom metric to use. This is the metric used for scheduler and earlyStopping. 
#' Needs to be a list with function `fun`, mode either `min` or `max` and a `name`,
#' `fun` needs to be a function that takes in prediction and labels and outputs a score.
#' @param seed seed to initialize weights of model with
#' @export
setEstimator <- function(learningRate='auto',
                         weightDecay = 0.0,
                         batchSize = 512,
                         epochs = 30,
                         device='cpu',
                         optimizer = torch$optim$AdamW,
                         scheduler = list(fun=torch$optim$lr_scheduler$ReduceLROnPlateau,
                                          params=list(patience=1)),
                         criterion = torch$nn$BCEWithLogitsLoss,
                         earlyStopping = list(useEarlyStopping=TRUE,
                                              params = list(patience=4)),
                         metric = "auc",
                         seed = NULL
) {
  
  checkIsClass(learningRate, c("numeric", "character"))
  if (inherits(learningRate, "character")) {
    if (learningRate != "auto"){
      stop(paste0('Learning rate should be either a numeric or "auto", you provided: ', learningRate))
    }
  }
  checkIsClass(weightDecay, "numeric")
  checkHigherEqual(weightDecay, 0.0)
  checkIsClass(batchSize, c("numeric", "integer"))
  checkHigher(batchSize, 0)
  checkIsClass(epochs, c("numeric", "integer"))
  checkHigher(epochs, 0)
  checkIsClass(device, c("character", "function"))
  checkIsClass(scheduler, "list")
  checkIsClass(earlyStopping, c("list", "NULL"))
  checkIsClass(metric, c("character", "list"))
  checkIsClass(seed, c("numeric", "integer", "NULL"))
  
  
  if (length(learningRate)==1 && learningRate=='auto') {findLR <- TRUE} else {findLR <- FALSE}
  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  estimatorSettings <- list(learningRate=learningRate,
                            weightDecay=weightDecay,
                            batchSize=batchSize,
                            epochs=epochs,
                            device=device,
                            earlyStopping=earlyStopping,
                            findLR=findLR,
                            metric=metric,
                            seed=seed[1])
  
  optimizer <- rlang::enquo(optimizer) 
  estimatorSettings$optimizer <- function() rlang::eval_tidy(optimizer)
  class(estimatorSettings$optimizer) <- c("delayed", class(estimatorSettings$optimizer))
  
  criterion <- rlang::enquo(criterion)
  estimatorSettings$criterion <- function() rlang::eval_tidy(criterion)
  class(estimatorSettings$criterion) <- c("delayed", class(estimatorSettings$criterion))
  
  scheduler <- rlang::enquo(scheduler)
  estimatorSettings$scheduler <- function() rlang::eval_tidy(scheduler)
  class(estimatorSettings$scheduler) <-c("delayed", class(estimatorSettings$scheduler))

  if (is.function(device)) {
    class(estimatorSettings$device) <- c("delayed",  class(estimatorSettings$device))
  }
  
  paramsToTune <- list()
  for (name in names(estimatorSettings)) {
    param <- estimatorSettings[[name]]
    if (length(param) > 1 && is.atomic(param)) {
      paramsToTune[[paste0('estimator.',name)]] <- param
    }
    if ("params" %in% names(param)) {
      for (name2 in names(param[["params"]])) {
        param2 <- param[["params"]][[name2]]
        if (length(param2) > 1) {
          paramsToTune[[paste0('estimator.',name,'.',name2)]] <- param2
        }
      }
    }
  }
  estimatorSettings$paramsToTune <- paramsToTune

  return(estimatorSettings)
}
 
#' fitEstimator
#'
#' @description
#' fits a deep learning estimator to data.
#'
#' @param trainData      the data to use
#' @param modelSettings  modelSettings object
#' @param analysisId     Id of the analysis
#' @param analysisPath   Path of the analysis
#' @param ...            Extra inputs
#'
#' @export
fitEstimator <- function(trainData,
                         modelSettings,
                         analysisId,
                         analysisPath,
                         ...) {
  start <- Sys.time()
  
  # check covariate data
  if (!FeatureExtraction::isCovariateData(trainData$covariateData)) {
    stop("Needs correct covariateData")
  }
  
  if (!is.null(trainData$folds)) {
    trainData$labels <- merge(trainData$labels, trainData$fold, by = "rowId")
  }
  mappedCovariateData <- PatientLevelPrediction::MapIds(
    covariateData = trainData$covariateData,
    cohort = trainData$labels
  )
  
  covariateRef <- mappedCovariateData$covariateRef
  
  outLoc <- PatientLevelPrediction::createTempModelLoc()
  cvResult <- do.call(
    what = gridCvDeep,
    args = list(
      mappedData = mappedCovariateData,
      labels = trainData$labels,
      modelSettings = modelSettings,
      modelLocation = outLoc,
      analysisPath = analysisPath
    )
  )
  
  hyperSummary <- do.call(rbind, lapply(cvResult$paramGridSearch, function(x) x$hyperSummary))
  prediction <- cvResult$prediction
  incs <- rep(1, covariateRef %>% dplyr::tally() %>% 
                dplyr::collect() %>%
                as.integer())
  covariateRef <- covariateRef %>%
    dplyr::arrange("columnId") %>%
    dplyr::collect() %>%
    dplyr::mutate(
      included = incs,
      covariateValue = 0,
      isNumeric = .data$columnId %in% cvResult$numericalIndex
    )
  
  comp <- start - Sys.time()
  result <- list(
    model = cvResult$estimator, # file.path(outLoc),
    
    preprocessing = list(
      featureEngineering = attr(trainData$covariateData, "metaData")$featureEngineering,
      tidyCovariates = attr(trainData$covariateData, "metaData")$tidyCovariateDataSettings,
      requireDenseMatrix = F
    ),
    prediction = prediction,
    modelDesign = PatientLevelPrediction::createModelDesign(
      targetId = attr(trainData, "metaData")$targetId,
      outcomeId = attr(trainData, "metaData")$outcomeId,
      restrictPlpDataSettings = attr(trainData, "metaData")$restrictPlpDataSettings,
      covariateSettings = attr(trainData, "metaData")$covariateSettings,
      populationSettings = attr(trainData, "metaData")$populationSettings,
      featureEngineeringSettings = attr(trainData$covariateData, "metaData")$featureEngineeringSettings,
      preprocessSettings = attr(trainData$covariateData, "metaData")$preprocessSettings,
      modelSettings = modelSettings,
      splitSettings = attr(trainData, "metaData")$splitSettings,
      sampleSettings = attr(trainData, "metaData")$sampleSettings
    ),
    trainDetails = list(
      analysisId = analysisId,
      analysisSource = "",
      developementDatabase = attr(trainData, "metaData")$cdmDatabaseSchema,
      attrition = attr(trainData, "metaData")$attrition,
      trainingTime = paste(as.character(abs(comp)), attr(comp, "units")),
      trainingDate = Sys.Date(),
      modelName = modelSettings$modelType,
      finalModelParameters = cvResult$finalParam,
      hyperParamSearch = hyperSummary
    ),
    covariateImportance = covariateRef
  )
  
  class(result) <- "plpModel"
  attr(result, "predictionFunction") <- "predictDeepEstimator"
  attr(result, "modelType") <- "binary"
  attr(result, "saveType") <- modelSettings$saveType
  
  return(result)
}

#' predictDeepEstimator
#'
#' @description
#' the prediction function for the estimator
#'
#' @param plpModel   the plpModel
#' @param data       plp data object or a torch dataset
#' @param cohort     data.frame with the rowIds of the people
#'
#' @export
predictDeepEstimator <- function(plpModel,
                                 data,
                                 cohort) {
  if (!"plpModel" %in% class(plpModel)) {
    plpModel <- list(model = plpModel)
    attr(plpModel, "modelType") <- "binary"
  }
  if ("plpData" %in% class(data)) {
    mappedData <- PatientLevelPrediction::MapIds(data$covariateData,
                                                 cohort = cohort,
                                                 mapping = plpModel$covariateImportance %>%
                                                   dplyr::select(
                                                     "columnId",
                                                     "covariateId"
                                                   )
    )
    data <- createDataset(mappedData, plpModel=plpModel)
  }
  
  # get predictions
  prediction <- cohort
  if (is.character(plpModel$model)) {
    modelSettings <- plpModel$modelDesign$modelSettings
    model <- torch$load(file.path(plpModel$model, "DeepEstimatorModel.pt"), map_location = "cpu")
    estimator <- createEstimator(modelType=modelSettings$modelType,
                                 modelParameters=model$model_parameters,
                                 estimatorSettings=model$estimator_settings)
    estimator$model$load_state_dict(model$model_state_dict)
    prediction$value <- estimator$predict_proba(data)
  } else {
    prediction$value <- plpModel$model$predict_proba(data)
  }
  
  attr(prediction, "metaData")$modelType <- attr(plpModel, "modelType")
  
  return(prediction)
}

#' gridCvDeep
#'
#' @description
#' Performs grid search for a deep learning estimator
#'
#'
#' @param mappedData    Mapped data with covariates
#' @param labels        Dataframe with the outcomes
#' @param modelSettings      Settings of the model
#' @param modelLocation Where to save the model
#' @param analysisPath  Path of the analysis
#'
#' @export
gridCvDeep <- function(mappedData,
                       labels,
                       modelSettings,
                       modelLocation,
                       analysisPath) {
  ParallelLogger::logInfo(paste0("Running hyperparameter search for ", modelSettings$modelType, " model"))
  
  ###########################################################################
  
  paramSearch <- modelSettings$param
  
  # TODO below chunk should be in a setupCache function
  trainCache <- TrainingCache$new(analysisPath)
    if (trainCache$isParamGridIdentical(paramSearch)) {
    gridSearchPredictons <- trainCache$getGridSearchPredictions()
  } else {
    gridSearchPredictons <- list()
    length(gridSearchPredictons) <- length(paramSearch)
    trainCache$saveGridSearchPredictions(gridSearchPredictons)
    trainCache$saveModelParams(paramSearch)
  }
  
  dataset <- createDataset(data=mappedData, labels=labels)
  
  fitParams <- names(paramSearch[[1]])[grepl("^estimator", names(paramSearch[[1]]))]
  findLR <- modelSettings$estimatorSettings$findLR
  for (gridId in trainCache$getLastGridSearchIndex():length(paramSearch)) {
    ParallelLogger::logInfo(paste0("Running hyperparameter combination no ", gridId))
    ParallelLogger::logInfo(paste0("HyperParameters: "))
    ParallelLogger::logInfo(paste(names(paramSearch[[gridId]]), paramSearch[[gridId]], collapse = " | "))
    currentModelParams <- paramSearch[[gridId]][modelSettings$modelParamNames]
    
    currentEstimatorSettings <- fillEstimatorSettings(modelSettings$estimatorSettings, fitParams, 
                                               paramSearch[[gridId]])
    
    # initiate prediction
    prediction <- NULL
    
    fold <- labels$index
    ParallelLogger::logInfo(paste0("Max fold: ", max(fold)))
    currentModelParams$catFeatures <- dataset$get_cat_features()$shape[[1]]
    currentModelParams$numFeatures <- dataset$get_numerical_features()$shape[[1]]
    if (findLR) {
      LrFinder <- createLRFinder(modelType = modelSettings$modelType,
                                 modelParameters = currentModelParams,
                                 estimatorSettings = currentEstimatorSettings
                                 )
      lr <- LrFinder$get_lr(dataset)
      ParallelLogger::logInfo(paste0("Auto learning rate selected as: ", lr))
      currentEstimatorSettings$learningRate <- lr
    }
    
    learnRates <- list()
    for (i in 1:max(fold)) {
      ParallelLogger::logInfo(paste0("Fold ", i))
      trainDataset <- torch$utils$data$Subset(dataset, indices = as.integer(which(fold != i) - 1)) # -1 for python 0-based indexing
      testDataset <- torch$utils$data$Subset(dataset, indices = as.integer(which(fold == i) -1)) # -1 for python 0-based indexing
    
      estimator <- createEstimator(modelType=modelSettings$modelType,
                                   modelParameters=currentModelParams,
                                   estimatorSettings=currentEstimatorSettings)
      estimator$fit(trainDataset, testDataset)
      
      ParallelLogger::logInfo("Calculating predictions on left out fold set...")
      
      prediction <- rbind(
        prediction,
        predictDeepEstimator(
          plpModel = estimator,
          data = testDataset,
          cohort = labels[fold == i, ]
        )
      )
      learnRates[[i]] <- list(
        LRs = estimator$learn_rate_schedule,
        bestEpoch = estimator$best_epoch
      )
    }
    maxIndex <- which.max(unlist(sapply(learnRates, `[`, 2)))
    paramSearch[[gridId]]$learnSchedule <- learnRates[[maxIndex]]
    
    gridSearchPredictons[[gridId]] <- list(
      prediction = prediction,
      param = paramSearch[[gridId]],
      gridPerformance = PatientLevelPrediction::computeGridPerformance(prediction, paramSearch[[gridId]])
    )
    
    # remove all predictions that are not the max performance
    indexOfMax <- which.max(unlist(lapply(gridSearchPredictons, function(x) x$gridPerformance$cvPerformance)))
    for (i in seq_along(gridSearchPredictons)) {
      if (!is.null(gridSearchPredictons[[i]])) {
        if (i != indexOfMax) {
          gridSearchPredictons[[i]]$prediction <- list(NULL)
        }
      }
    }
    ParallelLogger::logInfo(paste0("Caching all grid search results and prediction for best combination ", indexOfMax))
    trainCache$saveGridSearchPredictions(gridSearchPredictons)
  }
  
  paramGridSearch <- lapply(gridSearchPredictons, function(x) x$gridPerformance)
  
  # get best params
  indexOfMax <- which.max(unlist(lapply(gridSearchPredictons, function(x) x$gridPerformance$cvPerformance)))
  finalParam <- gridSearchPredictons[[indexOfMax]]$param
  
  # get best CV prediction
  cvPrediction <- gridSearchPredictons[[indexOfMax]]$prediction
  cvPrediction$evaluationType <- "CV"
  
  ParallelLogger::logInfo("Training final model using optimal parameters")
  # get the params
  modelParams <- finalParam[modelSettings$modelParamNames]
  
  
  # create the dir
  if (!dir.exists(file.path(modelLocation))) {
    dir.create(file.path(modelLocation), recursive = T)
  }
  
  modelParams$catFeatures <- dataset$get_cat_features()$shape[[1]]
  modelParams$numFeatures <- dataset$get_numerical_features()$shape[[1]]
  
  
  estimatorSettings <- fillEstimatorSettings(modelSettings$estimatorSettings, fitParams,
                                             finalParam)
  estimatorSettings$learningRate <- finalParam$learnSchedule$LRs[[1]]
  estimator <- createEstimator(modelType = modelSettings$modelType,
                               modelParameters = modelParams,
                               estimatorSettings = estimatorSettings)
 
  numericalIndex <- dataset$get_numerical_features()
  estimator$fit_whole_training_set(dataset, finalParam$learnSchedule$LRs)
  
  ParallelLogger::logInfo("Calculating predictions on all train data...")
  prediction <- predictDeepEstimator(
    plpModel = estimator,
    data = dataset,
    cohort = labels
  )
  prediction$evaluationType <- "Train"
  
  prediction <- rbind(
    prediction,
    cvPrediction
  )
  # modify prediction
  prediction <- prediction %>%
    dplyr::select(-"index")

  prediction$cohortStartDate <- as.Date(prediction$cohortStartDate, origin = "1970-01-01")
  
  
  # save torch code here
  estimator$save(modelLocation, "DeepEstimatorModel.pt")
  return(
    list(
      estimator = modelLocation,
      prediction = prediction,
      finalParam = finalParam,
      paramGridSearch = paramGridSearch,
      numericalIndex = numericalIndex$to_list()
    )
  )
}

# utility function to add instances of parameters to estimatorSettings during grid search
fillEstimatorSettings <- function(estimatorSettings, fitParams, paramSearch) {
  for (fp in fitParams) {
    components <- strsplit(fp, "[.]")[[1]]
    
    if (length(components)==2) {
      estimatorSettings[[components[[2]]]] <- paramSearch[[fp]]
    } else {
      estimatorSettings[[components[[2]]]]$params[[components[[3]]]] <- paramSearch[[fp]]
    }
  }
  return(estimatorSettings)
}

# utility function to evaluate any expressions or call functions passed as settings
evalEstimatorSettings <- function(estimatorSettings) {
  
  for (set in names(estimatorSettings)) {
    if (inherits(estimatorSettings[[set]], "delayed")) {
      estimatorSettings[[set]] <- estimatorSettings[[set]]()
    }
  }
  estimatorSettings
}

createEstimator <- function(modelType,
                            modelParameters,
                            estimatorSettings) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")

  Model <- reticulate::import_from_path(modelType, path=path)[[modelType]]
  
  Estimator <- reticulate::import_from_path("Estimator", path=path)$Estimator
  
  modelParameters <- camelCaseToSnakeCaseNames(modelParameters)
  estimatorSettings <- camelCaseToSnakeCaseNames(estimatorSettings)
  estimatorSettings <- evalEstimatorSettings(estimatorSettings)
  
  estimator <- Estimator(model = Model,
                         model_parameters = modelParameters,
                         estimator_settings = estimatorSettings)
  return(estimator)
}