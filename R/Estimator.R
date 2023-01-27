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
#' @param learningRate          what learning rate to use
#' @param weightDecay           what weight_decay to use
#' @param optimizer             which optimizer to use
#' @param scheduler             which learning rate scheduler to use
#' @param criterion             loss function to use
#' @param earlyStopping         If earlyStopping should be used which stops the training of your metric is not improving
#' @param earlyStoppingMetric   Which parameter to use for early stopping
#' @param patience              patience for earlyStopper
#' @param hyperparameterMetric  which metric to use for hyperparameter, loss, auc, auprc or a custom function
NULL

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

  param <- modelSettings$param

  # get the settings from the param
  settings <- attr(param, "settings")
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
      settings = settings,
      modelLocation = outLoc,
      paramSearch = param
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
      requireDenseMatrix = settings$requiresDenseMatrix
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
      modelName = settings$name,
      finalModelParameters = cvResult$finalParam,
      hyperParamSearch = hyperSummary
    ),
    covariateImportance = covariateRef
  )

  class(result) <- "plpModel"
  attr(result, "predictionFunction") <- "predictDeepEstimator"
  attr(result, "modelType") <- "binary"
  attr(result, "saveType") <- attr(param, "settings")$saveType

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
      modelType = attr(plpModel$modelDesign$modelSettings$param, "settings")$modelType,
      modelParameters = model$modelParameters,
      fitParameters = model$fitParameters,
      device = attr(plpModel$modelDesign$modelSettings$param, "settings")$device
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
#' @param settings      Settings of the model
#' @param modelLocation Where to save the model
#' @param paramSearch   model parameters to perform search over
#'
#' @export
gridCvDeep <- function(mappedData,
                       labels,
                       settings,
                       modelLocation,
                       paramSearch) {
  modelParamNames <- settings$modelParamNames
  fitParamNames <- c("weightDecay", "learningRate")
  epochs <- settings$epochs
  batchSize <- settings$batchSize
  modelType <- settings$modelType
  device <- settings$device

  ParallelLogger::logInfo(paste0("Running CV for ", modelType, " model"))

  ###########################################################################

  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- length(paramSearch)
  dataset <- Dataset(mappedData$covariates, labels$outcomeCount)
  for (gridId in 1:length(paramSearch)) {
    ParallelLogger::logInfo(paste0("Running hyperparameter combination no ", gridId))
    ParallelLogger::logInfo(paste0("HyperParameters: "))
    ParallelLogger::logInfo(paste(names(paramSearch[[gridId]]), paramSearch[[gridId]], collapse = " | "))
    modelParams <- paramSearch[[gridId]][modelParamNames]

    fitParams <- paramSearch[[gridId]][fitParamNames]
    fitParams$epochs <- epochs
    fitParams$batchSize <- batchSize


    # initiate prediction
    prediction <- c()

    fold <- labels$index
    ParallelLogger::logInfo(paste0("Max fold: ", max(fold)))
    modelParams$catFeatures <- dataset$numCatFeatures()
    modelParams$numFeatures <- dataset$numNumFeatures()
    learnRates <- list()
    for (i in 1:max(fold)) {
      ParallelLogger::logInfo(paste0("Fold ", i))
      trainDataset <- torch::dataset_subset(dataset, indices = which(fold != i))
      testDataset <- torch::dataset_subset(dataset, indices = which(fold == i))
      estimator <- Estimator$new(
        modelType = modelType,
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
  modelParams <- finalParam[modelParamNames]
  fitParams <- finalParam[fitParamNames]
  fitParams$epochs <- finalParam$learnSchedule$bestEpoch
  fitParams$batchSize <- batchSize
  # create the dir
  if (!dir.exists(file.path(modelLocation))) {
    dir.create(file.path(modelLocation), recursive = T)
  }
  modelParams$catFeatures <- dataset$numCatFeatures()
  modelParams$numFeatures <- dataset$numNumFeatures()

  estimator <- Estimator$new(
    modelType = modelType,
    modelParameters = modelParams,
    fitParameters = fitParams,
    device = device
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