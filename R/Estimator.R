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
                         optimizer = torchopt::optim_adamw,
                         scheduler = list(fun=torch::lr_reduce_on_plateau,
                                          params=list(patience=1)),
                         criterion = torch::nn_bce_with_logits_loss,
                         earlyStopping = list(useEarlyStopping=TRUE,
                                              params = list(patience=4)),
                         metric = "auc",
                         seed = NULL
) {
  if (length(learningRate)==1 && learningRate=='auto') {findLR <- TRUE} else {findLR <- FALSE}
  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  
  
  estimatorSettings <- list(learningRate=learningRate,
                            weightDecay=weightDecay,
                            batchSize=batchSize,
                            epochs=epochs,
                            device=device,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            criterion=criterion,
                            earlyStopping=earlyStopping,
                            findLR=findLR,
                            metric=metric,
                            seed=seed[1] 
  )
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
#' @param ...            Extra inputs
#'
#' @export
fitEstimator <- function(trainData,
                         modelSettings,
                         analysisId,
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
      modelLocation = outLoc
    )
  )
  
  hyperSummary <- do.call(rbind, lapply(cvResult$paramGridSearch, function(x) x$hyperSummary))
  prediction <- cvResult$prediction
  incs <- rep(1, covariateRef %>% dplyr::tally() %>% 
                dplyr::collect ()
              %>% dplyr::pull())
  covariateRef <- covariateRef %>%
    dplyr::collect() %>%
    dplyr::mutate(
      included = incs,
      covariateValue = 0,
      isNumeric = cvResult$numericalIndex
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
    data <- Dataset(mappedData$covariates,
                    numericalIndex = plpModel$covariateImportance$isNumeric
    )
  }
  
  # get predictions
  prediction <- cohort
  if (is.character(plpModel$model)) {
    model <- torch::torch_load(file.path(plpModel$model, "DeepEstimatorModel.pt"), device = "cpu")
    estimator <- Estimator$new(
      modelType = plpModel$modelDesign$modelSettings$modelType,
      modelParameters = model$modelParameters,
      estimatorSettings = model$estimatorSettings
    )
    estimator$model$load_state_dict(model$modelStateDict)
    prediction$value <- estimator$predictProba(data)
  } else {
    prediction$value <- plpModel$model$predictProba(data)
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
#'
#' @export
gridCvDeep <- function(mappedData,
                       labels,
                       modelSettings,
                       modelLocation) {
  ParallelLogger::logInfo(paste0("Running hyperparameter search for ", modelSettings$modelType, " model"))
  
  ###########################################################################
  
  paramSearch <- modelSettings$param
  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- length(paramSearch)
  dataset <- Dataset(mappedData$covariates, labels$outcomeCount)
  
  estimatorSettings <- modelSettings$estimatorSettings
  
  fitParams <- names(paramSearch[[1]])[grepl("^estimator", names(paramSearch[[1]]))]
  
  for (gridId in 1:length(paramSearch)) {
    ParallelLogger::logInfo(paste0("Running hyperparameter combination no ", gridId))
    ParallelLogger::logInfo(paste0("HyperParameters: "))
    ParallelLogger::logInfo(paste(names(paramSearch[[gridId]]), paramSearch[[gridId]], collapse = " | "))
    modelParams <- paramSearch[[gridId]][modelSettings$modelParamNames]
    
    
    estimatorSettings <- fillEstimatorSettings(estimatorSettings, fitParams, 
                                               paramSearch[[gridId]])

    # initiate prediction
    prediction <- c()
    
    fold <- labels$index
    ParallelLogger::logInfo(paste0("Max fold: ", max(fold)))
    modelParams$catFeatures <- dataset$numCatFeatures()
    modelParams$numFeatures <- dataset$numNumFeatures()
    
    if (estimatorSettings$findLR) {
      lr <- lrFinder(dataset=dataset, 
                     modelType = modelSettings$modelType,
                     modelParams = modelParams,
                     estimatorSettings = estimatorSettings)
      ParallelLogger::logInfo(paste0("Auto learning rate selected as: ", lr))
      estimatorSettings$learningRate <- lr
    }
    
    
    learnRates <- list()
    for (i in 1:max(fold)) {
      ParallelLogger::logInfo(paste0("Fold ", i))
      trainDataset <- torch::dataset_subset(dataset, indices = which(fold != i))
      testDataset <- torch::dataset_subset(dataset, indices = which(fold == i))
      estimator <- Estimator$new(
        modelType = modelSettings$modelType,
        modelParameters = modelParams,
        estimatorSettings = estimatorSettings
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
          cohort = labels[fold == i, ]
        )
      )
      learnRates[[i]] <- list(
        LRs = estimator$learnRateSchedule,
        bestEpoch = estimator$bestEpoch
      )
    }
    maxIndex <- which.max(unlist(sapply(learnRates, `[`, 2)))
    paramSearch[[gridId]]$learnSchedule <- learnRates[[maxIndex]]
    
    gridSearchPredictons[[gridId]] <- list(
      prediction = prediction,
      param = paramSearch[[gridId]]
    )
  }
  # get best para (this could be modified to enable any metric instead of AUC, just need metric input in function)
  paramGridSearch <- lapply(gridSearchPredictons, function(x) {
    do.call(PatientLevelPrediction::computeGridPerformance, x)
  }) # cvAUCmean, cvAUC, param
  
  optimalParamInd <- which.max(unlist(lapply(paramGridSearch, function(x) x$cvPerformance)))
  finalParam <- paramGridSearch[[optimalParamInd]]$param
  
  cvPrediction <- gridSearchPredictons[[optimalParamInd]]$prediction
  cvPrediction$evaluationType <- "CV"
  
  ParallelLogger::logInfo("Training final model using optimal parameters")
  # get the params
  modelParams <- finalParam[modelSettings$modelParamNames]
  
  
  # create the dir
  if (!dir.exists(file.path(modelLocation))) {
    dir.create(file.path(modelLocation), recursive = T)
  }
  modelParams$catFeatures <- dataset$numCatFeatures()
  modelParams$numFeatures <- dataset$numNumFeatures()
  
  estimatorSettings <- fillEstimatorSettings(estimatorSettings, fitParams,
                                             finalParam)
  
  estimator <- Estimator$new(
    modelType = modelSettings$modelType,
    modelParameters = modelParams,
    estimatorSettings = estimatorSettings
  )
  numericalIndex <- dataset$getNumericalIndex()
  
  estimator$fitWholeTrainingSet(dataset, finalParam$learnSchedule$LRs)
  
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
      numericalIndex = numericalIndex
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