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
#' @param device  what device to train on, can be a string or a function to
#' that evaluates to the device during runtime
#' @param optimizer which optimizer to use
#' @param scheduler which learning rate scheduler to use
#' @param criterion loss function to use
#' @param earlyStopping If earlyStopping should be used which stops the
#' training of your metric is not improving
#' @param compile if the model should be compiled before training, default FALSE
#' @param metric either `auc` or `loss` or a custom metric to use. This is the
#' metric used for scheduler and earlyStopping.
#' Needs to be a list with function `fun`, mode either `min` or `max` and a
#' `name`,
#' `fun` needs to be a function that takes in prediction and labels and
#' outputs a score.
#' @param accumulationSteps how many steps to accumulate gradients before
#' updating weights, can also be a function that is evaluated during runtime
#' @param seed seed to initialize weights of model with
#' @param trainValidationSplit if TRUE, perform a train-validation split for
#' model selection instead of cross validation
#' @export
setEstimator <- function(
    learningRate = "auto",
    weightDecay = 0.0,
    batchSize = 512,
    epochs = 30,
    device = "cpu",
    optimizer = torch$optim$AdamW,
    scheduler = list(
      fun = torch$optim$lr_scheduler$ReduceLROnPlateau,
      params = list(patience = 1)
    ),
    criterion = torch$nn$BCEWithLogitsLoss,
    earlyStopping = list(
      useEarlyStopping = TRUE,
      params = list(patience = 4)
    ),
    compile = FALSE,
    metric = "auc",
    accumulationSteps = NULL,
    seed = NULL,
    trainValidationSplit = FALSE) {
  checkIsClass(learningRate, c("numeric", "character"))
  if (inherits(learningRate, "character") && learningRate != "auto") {
    stop(paste0(
      'Learning rate should be either a numeric or "auto", you provided: ',
      learningRate
    ))
  }
  checkIsClass(weightDecay, "numeric")
  checkHigherEqual(weightDecay, 0.0)
  checkIsClass(batchSize, c("numeric", "integer"))
  checkHigher(batchSize, 0)
  checkIsClass(epochs, c("numeric", "integer"))
  checkHigher(epochs, 0)
  checkIsClass(earlyStopping, c("list", "NULL"))
  checkIsClass(compile, "logical")
  checkIsClass(metric, c("character", "list"))
  checkIsClass(seed, c("numeric", "integer", "NULL"))
  checkIsClass(trainValidationSplit, "logical")
  if (!is.null(accumulationSteps) && !is.function(accumulationSteps)) {
    checkHigher(accumulationSteps, 0)
    checkIsClass(accumulationSteps, c("numeric", "integer"))
    if (batchSize %% accumulationSteps != 0) {
      stop("Batch size should be divisible by accumulation steps")
    }
  }

  if (length(learningRate) == 1 && learningRate == "auto") {
    findLR <- TRUE
  } else {
    findLR <- FALSE
  }
  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  estimatorSettings <- list(
    learningRate = learningRate,
    weightDecay = weightDecay,
    batchSize = batchSize,
    epochs = epochs,
    device = device,
    earlyStopping = earlyStopping,
    compile = compile,
    findLR = findLR,
    metric = metric,
    accumulationSteps = accumulationSteps,
    seed = seed[1]
  )
  if (!is.null(trainValidationSplit)) {
    estimatorSettings$trainValidationSplit <- trainValidationSplit
  }
  optimizer <- rlang::enquo(optimizer)
  estimatorSettings$optimizer <- function() rlang::eval_tidy(optimizer)
  class(estimatorSettings$optimizer) <- c(
    "delayed",
    class(estimatorSettings$optimizer)
  )

  criterion <- rlang::enquo(criterion)
  estimatorSettings$criterion <- function() rlang::eval_tidy(criterion)
  class(estimatorSettings$criterion) <- c(
    "delayed",
    class(estimatorSettings$criterion)
  )

  scheduler <- rlang::enquo(scheduler)
  estimatorSettings$scheduler <- function() rlang::eval_tidy(scheduler)
  class(estimatorSettings$scheduler) <- c(
    "delayed",
    class(estimatorSettings$scheduler)
  )

  if (is.function(device)) {
    class(estimatorSettings$device) <- c(
      "delayed",
      class(estimatorSettings$device)
    )
  }
  if (is.function(accumulationSteps)) {
    class(estimatorSettings$accumulationSteps) <- c(
      "delayed",
      class(estimatorSettings$accumulationSteps)
    )
  }

  estimatorSettings$paramsToTune <- extractParamsToTune(estimatorSettings)
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
#' @param hyperparameterSettings PLP hyperparameter settings
#' @param ...            Extra inputs
#'
#' @export
fitEstimator <- function(
    trainData,
    modelSettings,
    analysisId,
    analysisPath,
    hyperparameterSettings = NULL,
    ...) {
  fitDeepPlpClassifier(
    trainData = trainData,
    modelSettings = modelSettings,
    hyperparameterSettings = hyperparameterSettings,
    analysisId = analysisId,
    analysisPath = analysisPath,
    ...
  )
}

#' fitDeepPlpClassifier
#'
#' @description
#' Fits a deep learning estimator using PLP hyperparameter settings and
#' DeepPLP's cache-aware training loop.
#'
#' @param trainData the data to use
#' @param modelSettings modelSettings object
#' @param hyperparameterSettings PLP hyperparameter settings
#' @param analysisId Id of the analysis
#' @param analysisPath Path of the analysis
#' @param ... Extra inputs
#'
#' @export
fitDeepPlpClassifier <- function(
    trainData,
    modelSettings,
    hyperparameterSettings = NULL,
    analysisId,
    analysisPath,
    ...) {
  start <- Sys.time()
  if (!is.null(trainData$folds)) {
    trainData$labels <- merge(trainData$labels, trainData$fold, by = "rowId")
  }

  if (attr(modelSettings$param, "settings")$modelType == "Finetuner") {
    # make sure to use same mapping from covariateIds to columns if finetuning
    path <- modelSettings$param[[1]]$modelPath
    oldCovImportance <- utils::read.csv(
      file.path(
        path,
        "covariateImportance.csv"
      )
    )
    mapping <- oldCovImportance %>% dplyr::select("columnId", "covariateId")
    mappedCovariateData <- PatientLevelPrediction::MapIds(
      covariateData = trainData$covariateData,
      cohort = trainData$labels,
      mapping = mapping
    )
  } else {
    mappedCovariateData <- PatientLevelPrediction::MapIds(
      covariateData = trainData$covariateData,
      cohort = trainData$labels
    )
  }

  outLoc <- PatientLevelPrediction::createTempModelLoc()
  cvResult <- do.call(
    what = gridCvDeep,
    args = list(
      mappedData = mappedCovariateData,
      labels = trainData$labels,
      modelSettings = modelSettings,
      hyperparameterSettings = hyperparameterSettings,
      modelLocation = outLoc,
      analysisPath = analysisPath
    )
  )

  hyperSummary <- do.call(
    rbind,
    lapply(
      cvResult$paramGridSearch,
      function(x) x$hyperSummary
    )
  )
  prediction <- cvResult$prediction
  covariateRef <- py_to_r(cvResult[["featureInfo"]]$data_reference)
  incs <- rep(
    1,
    covariateRef %>%
      dplyr::tally() %>%
      dplyr::collect() %>%
      as.integer()
  )
  covariateRef <- covariateRef %>%
    dplyr::arrange(.data$columnId) %>%
    dplyr::collect() %>%
    dplyr::mutate(
      included = incs,
      covariateValue = 0
    ) %>%
    normalizeCovariateReferenceTypes()

  comp <- start - Sys.time()
  modelSettings$estimatorSettings$initStrategy <- NULL
  result <- list(
    model = cvResult$estimator,
    preprocessing = list(
      featureEngineering = attr(
        trainData$covariateData,
        "metaData"
      )$featureEngineering,
      tidyCovariates = attr(
        trainData$covariateData,
        "metaData"
      )$tidyCovariateDataSettings,
      requireDenseMatrix = FALSE
    ),
    prediction = prediction,
    modelDesign = PatientLevelPrediction::createModelDesign(
      targetId = attr(trainData, "metaData")$targetId,
      outcomeId = attr(trainData, "metaData")$outcomeId,
      restrictPlpDataSettings = attr(
        trainData,
        "metaData"
      )$restrictPlpDataSettings,
      covariateSettings = attr(trainData, "metaData")$covariateSettings,
      populationSettings = attr(trainData, "metaData")$populationSettings,
      featureEngineeringSettings = attr(
        trainData$covariateData,
        "metaData"
      )$featureEngineeringSettings,
      preprocessSettings = attr(
        trainData$covariateData,
        "metaData"
      )$preprocessSettings,
      modelSettings = modelSettings,
      hyperparameterSettings = cvResult$hyperparameterSettings,
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
  attr(result, "predictionFunction") <-
    "DeepPatientLevelPrediction::predictDeepEstimator"
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
predictDeepEstimator <- function(plpModel, data, cohort) {
  if (!"plpModel" %in% class(plpModel)) {
    plpModel <- list(model = plpModel)
    attr(plpModel, "modelType") <- "binary"
  }

  if (!is.null(plpModel$covariateImportance)) {
    # this means that the model finished training since only in the end
    # covariateImportance is added
    mappedData <- PatientLevelPrediction::MapIds(
      data$covariateData,
      cohort = cohort,
      mapping = plpModel$covariateImportance %>%
        dplyr::select("columnId", "covariateId")
    )
    data <- createDataset(mappedData,
      plpModel = plpModel,
      temporalSettings = attr(plpModel$modelDesign$modelSettings$param, "temporalSettings")
    )
  } else if ("plpData" %in% class(data)) {
    mappedData <- PatientLevelPrediction::MapIds(
      data$covariateData,
      cohort = cohort,
      mapping = plpModel$covariateImportance %>%
        dplyr::select("columnId", "covariateId")
    )
    data <- createDataset(mappedData,
      plpModel = plpModel
    )
  }

  # get predictions
  prediction <- cohort
  if (is.character(plpModel$model)) {
    model <- torch$load(
      file.path(plpModel$model, "DeepEstimatorModel.pt"),
      map_location = "cpu",
      weights_only = FALSE
    )
    if (is.null(model$model_parameters$model_type)) {
      # for backwards compatibility
      model$model_parameters$model_type <- plpModel$modelDesign$modelSettings$modelType
    }
    model$estimator_settings$device <-
      plpModel$modelDesign$modelSettings$estimatorSettings$device
    modelParameters <- snakeCaseToCamelCaseNames(model$model_parameters)
    estimatorSettings <- snakeCaseToCamelCaseNames(model$estimator_settings)
    parameters <- list(
      modelParameters = modelParameters,
      estimatorSettings = estimatorSettings
    )
    estimator <-
      createEstimator(parameters = parameters)
    estimator$model$load_state_dict(model$model_state_dict)
    prediction$value <- estimator$predict_proba(data)
  } else {
    prediction$value <- plpModel$model$predict_proba(data)
  }
  prediction$value <- as.numeric(prediction$value)
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
#' @param hyperparameterSettings PLP hyperparameter settings
#' @param modelLocation Where to save the model
#' @param analysisPath  Path of the analysis
#'
#' @export
gridCvDeep <- function(
    mappedData,
    labels,
    modelSettings,
    hyperparameterSettings = NULL,
    modelLocation,
    analysisPath) {
  ParallelLogger::logInfo(paste0(
    "Running hyperparameter search for ",
    modelSettings$modelType,
    " model"
  ))
  ###########################################################################

  hyperparameterSettings <- resolveHyperparameterSettings(
    modelSettings = modelSettings,
    hyperparameterSettings = hyperparameterSettings
  )
  tuningMetric <- hyperparameterSettings$tuningMetric
  paramDefinition <- modelSettings$paramDefinition %||% modelSettings$param
  if (isAdaptiveHyperparameterSettings(hyperparameterSettings)) {
    return(gridCvDeepAdaptive(
      mappedData = mappedData,
      labels = labels,
      modelSettings = modelSettings,
      hyperparameterSettings = hyperparameterSettings,
      modelLocation = modelLocation,
      analysisPath = analysisPath,
      paramDefinition = paramDefinition
    ))
  }
  paramSearch <- createCandidatePool(
    paramDefinition = paramDefinition,
    hyperparameterSettings = hyperparameterSettings
  )

  # setup cache for hyperparameterResults
  trainCache <- setupSearchCache(
    analysisPath = analysisPath,
    paramDefinition = paramDefinition,
    hyperparameterSettings = hyperparameterSettings,
    candidatePool = paramSearch
  )
  paramSearch <- trainCache$getCandidatePool() %||% paramSearch
  hyperparameterResults <- trainCache$getSearchResults()
  dataset <- createDataset(
    data = mappedData,
    labels = labels,
    temporalSettings = attr(paramDefinition, "temporalSettings")
  )

  if (!trainCache$isSearchFull()) {
    startIndex <- trainCache$getNextSearchIndex()
    if (startIndex <= length(paramSearch)) {
      for (gridId in startIndex:length(paramSearch)) {
        parameters <- postProcessHyperParameters(
          paramSearch[[gridId]],
          modelSettings
        )
      ParallelLogger::logInfo(paste0(
        "Running hyperparameter combination no ",
        gridId
      ))
      ParallelLogger::logInfo(paste0("HyperParameters: "))
      ParallelLogger::logInfo(paste(
        names(parameters),
        parameters,
        collapse = " | "
      ))
      hyperparameterResults[[gridId]] <-
        doCrossValidation(
          dataset,
          labels = labels,
          parameters = parameters,
          modelSettings = modelSettings,
          tuningMetric = tuningMetric
        )
      # remove all predictions that are not the max performance
      hyperparameterResults <- trainCache$trimPerformance(
        hyperparameterResults,
        maximize = isTRUE(tuningMetric$maximize)
      )
      trainCache$saveSearchResults(hyperparameterResults)
      }
    }
  }
  paramGridSearch <- lapply(
    hyperparameterResults,
    function(x) x$gridPerformance
  )
  # get best params
  indexOfMax <-
    selectBestHyperparameterIndex(unlist(lapply(
      hyperparameterResults,
      function(x) x$gridPerformance$cvPerformance
    )), maximize = isTRUE(tuningMetric$maximize))
  if (length(indexOfMax) == 0) {
    stop("No hyperparameter combination has valid results")
  }
  finalParam <- hyperparameterResults[[indexOfMax]]$param

  paramGridSearch <- lapply(
    hyperparameterResults,
    function(x) x$gridPerformance
  )
  # get best CV prediction
  cvPrediction <- hyperparameterResults[[indexOfMax]]$prediction
  if (modelSettings$estimatorSettings$trainValidationSplit) {
    cvPrediction$evaluationType <- "Validation"
  } else {
    cvPrediction$evaluationType <- "CV"
  }

  ParallelLogger::logInfo("Training final model using optimal parameters")
  trainPrediction <- trainFinalModel(dataset, finalParam, modelSettings, labels)
  prediction <- rbind(
    trainPrediction$prediction,
    cvPrediction
  )
  # remove fold index from predictions and remove cohortStartDate
  prediction <- prediction %>%
    dplyr::select(-"index")
  prediction$cohortStartDate <- as.Date(
    prediction$cohortStartDate,
    origin = "1970-01-01"
  )
  featureInfo <- dataset$get_feature_info()

  # save torch code here
  if (!dir.exists(file.path(modelLocation))) {
    dir.create(file.path(modelLocation), recursive = TRUE)
  }
  trainPrediction$estimator$save(modelLocation, "DeepEstimatorModel.pt")
  return(
    list(
      estimator = modelLocation,
      prediction = prediction,
      finalParam = finalParam,
      paramGridSearch = paramGridSearch,
      featureInfo = featureInfo,
      hyperparameterSettings = hyperparameterSettings
    )
  )
}

resolveHyperparameterSettings <- function(modelSettings, hyperparameterSettings) {
  if (is.null(hyperparameterSettings) ||
    isDefaultHyperparameterSettings(hyperparameterSettings)) {
    return(
      modelSettings$legacyHyperparameterSettings %||%
        PatientLevelPrediction::createHyperparameterSettings()
    )
  }

  if (isTRUE(modelSettings$legacySearchExplicit)) {
    warning(
      paste(
        "Ignoring deprecated helper-level hyperparameter search settings",
        "because `hyperparameterSettings` was supplied."
      ),
      call. = FALSE
    )
  }
  hyperparameterSettings
}

isDefaultHyperparameterSettings <- function(hyperparameterSettings) {
  if (!inherits(hyperparameterSettings, "hyperparameterSettings")) {
    return(FALSE)
  }
  identical(hyperparameterSettings$search, "grid") &&
    is.null(hyperparameterSettings$sampleSize) &&
    is.null(hyperparameterSettings$randomSeed) &&
    is.null(hyperparameterSettings$generator) &&
    identical(hyperparameterSettings$tuningMetric$name, "AUC") &&
    isTRUE(hyperparameterSettings$tuningMetric$maximize)
}

createCandidatePool <- function(paramDefinition, hyperparameterSettings) {
  if (identical(hyperparameterSettings$search, "custom") &&
    !is.function(hyperparameterSettings$generator)) {
    stop(
      paste(
        "DeepPLP requires custom hyperparameter generators to return a",
        "serializable candidate pool for cached training in this release."
      ),
      call. = FALSE
    )
  }
  prepareHyperparameterGrid <- utils::getFromNamespace(
    "prepareHyperparameterGrid",
    "PatientLevelPrediction"
  )
  iterator <- if (!is.null(hyperparameterSettings$randomSeed)) {
    withr::with_seed(
      hyperparameterSettings$randomSeed,
      prepareHyperparameterGrid(paramDefinition, hyperparameterSettings)
    )
  } else {
    prepareHyperparameterGrid(paramDefinition, hyperparameterSettings)
  }
  candidatePool <- list()
  history <- list()
  repeat {
    candidate <- iterator$getNext(history)
    if (is.null(candidate)) {
      break
    }
    candidatePool[[length(candidatePool) + 1L]] <- candidate
    history[[length(history) + 1L]] <- list(param = candidate)
  }
  iterator$finalize(history)
  candidatePool
}

isAdaptiveHyperparameterSettings <- function(hyperparameterSettings) {
  identical(hyperparameterSettings$search, "custom") &&
    !is.null(hyperparameterSettings$generator) &&
    !is.function(hyperparameterSettings$generator)
}

gridCvDeepAdaptive <- function(
    mappedData,
    labels,
    modelSettings,
    hyperparameterSettings,
    modelLocation,
    analysisPath,
    paramDefinition) {
  generator <- hyperparameterSettings$generator
  if (!is.function(generator$saveState) ||
    !is.function(generator$loadState)) {
    stop(
      paste(
        "DeepPLP requires adaptive custom hyperparameter generators to",
        "provide saveState() and loadState(state) methods so cached training",
        "can resume safely."
      ),
      call. = FALSE
    )
  }

  trainCache <- setupAdaptiveSearchCache(
    analysisPath = analysisPath,
    paramDefinition = paramDefinition,
    hyperparameterSettings = hyperparameterSettings
  )
  dataset <- createDataset(
    data = mappedData,
    labels = labels,
    temporalSettings = attr(paramDefinition, "temporalSettings")
  )
  history <- trainCache$getSearchHistory()
  hyperparameterResults <- trainCache$getSearchResults() %||% list()
  if (!trainCache$isAdaptiveSearchComplete()) {
    generatorState <- trainCache$getGeneratorState()
    if (!is.null(generatorState)) {
      generator$loadState(generatorState)
    } else if (is.function(generator$initialize)) {
      generator$initialize(
        definition = paramDefinition,
        settings = hyperparameterSettings
      )
    }
    repeat {
      candidate <- generator$getNext(history)
      if (is.null(candidate)) {
        break
      }
      gridId <- length(hyperparameterResults) + 1L
      parameters <- postProcessHyperParameters(candidate, modelSettings)
      ParallelLogger::logInfo(paste0(
        "Running adaptive hyperparameter combination no ",
        gridId
      ))
      hyperparameterResults[[gridId]] <- doCrossValidation(
        dataset,
        labels = labels,
        parameters = parameters,
        modelSettings = modelSettings,
        tuningMetric = hyperparameterSettings$tuningMetric
      )
      history[[gridId]] <- hyperparameterResults[[gridId]]$gridPerformance
      hyperparameterResults <- trainCache$trimPerformance(
        hyperparameterResults,
        maximize = isTRUE(hyperparameterSettings$tuningMetric$maximize)
      )
      trainCache$saveAdaptiveSearchState(
        inSearchResults = hyperparameterResults,
        inHistory = history,
        inGeneratorState = generator$saveState(),
        complete = FALSE
      )
    }
    finalizeFn <- generator$finalize %||% function(...) invisible(NULL)
    finalizeFn(history)
    trainCache$saveAdaptiveSearchState(
      inSearchResults = hyperparameterResults,
      inHistory = history,
      inGeneratorState = generator$saveState(),
      complete = TRUE
    )
  }
  finishGridCvDeep(
    dataset = dataset,
    labels = labels,
    modelSettings = modelSettings,
    modelLocation = modelLocation,
    hyperparameterResults = hyperparameterResults,
    hyperparameterSettings = hyperparameterSettings
  )
}

postProcessHyperParameters <- function(parameters, modelSettings) {
  postProcess <- modelSettings$postProcessHyperParameters
  if (is.function(postProcess)) {
    parameters <- postProcess(parameters)
  }
  parameters
}

selectBestHyperparameterIndex <- function(values, maximize = TRUE) {
  values <- as.numeric(values)
  if (all(is.na(values))) {
    return(integer())
  }
  if (isTRUE(maximize)) {
    which.max(values)
  } else {
    which.min(values)
  }
}

finishGridCvDeep <- function(
    dataset,
    labels,
    modelSettings,
    modelLocation,
    hyperparameterResults,
    hyperparameterSettings) {
  paramGridSearch <- lapply(
    hyperparameterResults,
    function(x) x$gridPerformance
  )
  indexOfMax <- selectBestHyperparameterIndex(unlist(lapply(
    hyperparameterResults,
    function(x) x$gridPerformance$cvPerformance
  )), maximize = isTRUE(hyperparameterSettings$tuningMetric$maximize))
  if (length(indexOfMax) == 0) {
    stop("No hyperparameter combination has valid results")
  }
  finalParam <- hyperparameterResults[[indexOfMax]]$param
  cvPrediction <- hyperparameterResults[[indexOfMax]]$prediction
  if (modelSettings$estimatorSettings$trainValidationSplit) {
    cvPrediction$evaluationType <- "Validation"
  } else {
    cvPrediction$evaluationType <- "CV"
  }

  ParallelLogger::logInfo("Training final model using optimal parameters")
  trainPrediction <- trainFinalModel(dataset, finalParam, modelSettings, labels)
  prediction <- rbind(
    trainPrediction$prediction,
    cvPrediction
  )
  prediction <- prediction %>%
    dplyr::select(-"index")
  prediction$cohortStartDate <- as.Date(
    prediction$cohortStartDate,
    origin = "1970-01-01"
  )
  featureInfo <- dataset$get_feature_info()

  if (!dir.exists(file.path(modelLocation))) {
    dir.create(file.path(modelLocation), recursive = TRUE)
  }
  trainPrediction$estimator$save(modelLocation, "DeepEstimatorModel.pt")
  list(
    estimator = modelLocation,
    prediction = prediction,
    finalParam = finalParam,
    paramGridSearch = paramGridSearch,
    featureInfo = featureInfo,
    hyperparameterSettings = hyperparameterSettings
  )
}

# utility function to add instances of parameters to estimatorSettings
# during grid search
fillEstimatorSettings <- function(estimatorSettings, fitParams, paramSearch) {
  for (fp in fitParams) {
    components <- strsplit(fp, "[.]")[[1]]

    if (length(components) == 2) {
      estimatorSettings[[components[[2]]]] <- paramSearch[[fp]]
    } else {
      estimatorSettings[[components[[2]]]]$params[[components[[3]]]] <-
        paramSearch[[fp]]
    }
  }
  return(estimatorSettings)
}

# utility function to evaluate any expressions or call functions passed as
# settings
evalEstimatorSettings <- function(estimatorSettings) {
  for (set in names(estimatorSettings)) {
    if (inherits(estimatorSettings[[set]], "delayed")) {
      estimatorSettings[[set]] <- estimatorSettings[[set]]()
    }
  }
  estimatorSettings
}

createEstimator <- function(parameters) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  model <- reticulate::import_from_path(
    parameters$modelParameters$modelType,
    path = path
  )[[parameters$modelParameters$modelType]]

  estimator <- reticulate::import_from_path("Estimator", path = path)$Estimator

  parameters$modelParameters <- camelCaseToSnakeCaseNames(
    parameters$modelParameters
  )
  parameters$estimatorSettings <- camelCaseToSnakeCaseNames(
    parameters$estimatorSettings
  )
  parameters$estimatorSettings <- evalEstimatorSettings(
    parameters$estimatorSettings
  )
  parameters <- camelCaseToSnakeCaseNames(parameters)
  estimator <- estimator(
    model = model,
    parameters = parameters
  )
  return(estimator)
}

#' Train one DeepPLP model candidate
#'
#' @description
#' Callback matching the PLP 6.6 model interface.
#'
#' @param dataMatrix Torch dataset or subset
#' @param labels Cohort labels
#' @param hyperParameters Candidate hyperparameters
#' @param settings DeepPLP model settings
#'
#' @export
trainDeepPlpCandidate <- function(
    dataMatrix,
    labels,
    hyperParameters,
    settings) {
  fitParams <- names(hyperParameters)[grepl(
    "^estimator",
    names(hyperParameters)
  )]
  modelParamNames <- setdiff(
    names(hyperParameters)[!grepl("^estimator", names(hyperParameters))],
    "learnSchedule"
  )
  currentModelParams <- hyperParameters[modelParamNames]
  currentModelParams$modelType <- settings$deepModelType %||%
    settings$modelName %||%
    settings$modelType
  currentModelParams$feature_info <- getDeepFeatureInfo(dataMatrix)
  currentEstimatorSettings <- fillEstimatorSettings(
    settings$estimatorSettings,
    fitParams,
    hyperParameters
  )
  currentParameters <- list(
    modelParameters = currentModelParams,
    estimatorSettings = currentEstimatorSettings
  )
  estimator <- createEstimator(currentParameters)
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  fit_estimator <- reticulate::import_from_path(
    "Estimator",
    path = path
  )$fit_estimator
  fit_estimator(estimator, dataMatrix, dataMatrix)
  estimator
}

getDeepFeatureInfo <- function(dataMatrix) {
  if (reticulate::py_has_attr(dataMatrix, "get_feature_info")) {
    return(dataMatrix$get_feature_info())
  }
  if (
    reticulate::py_has_attr(dataMatrix, "dataset") &&
      reticulate::py_has_attr(dataMatrix$dataset, "get_feature_info")
  ) {
    return(dataMatrix$dataset$get_feature_info())
  }
  stop("Unable to get feature information from the data matrix.", call. = FALSE)
}

#' Get DeepPLP variable importance
#'
#' @description
#' Placeholder callback for the PLP 6.6 model interface. DeepPLP constructs
#' covariate importance from the mapped feature reference after fitting.
#'
#' @param plpModel A fitted PLP model
#'
#' @export
getDeepVariableImportance <- function(plpModel) {
  plpModel$covariateImportance %||% data.frame()
}

doCrossValidation <- function(
    dataset,
    labels,
    parameters,
    modelSettings,
    tuningMetric = NULL) {
  tuningMetric <- tuningMetric %||%
    PatientLevelPrediction::createHyperparameterSettings()$tuningMetric
  crossValidationResults <-
    tryCatch(
      doCrossValidationImpl(
        dataset,
        labels,
        parameters,
        modelSettings,
        tuningMetric
      ),
      error = function(e) {
        if (inherits(e, "torch.cuda.OutOfMemoryError")) {
          ParallelLogger::logError(
            "Out of memory error during cross validation, trying to continue
            with next hyperparameter combination"
          )
          crossValidationResults <- list()
          crossValidationResults$prediction <- labels
          crossValidationResults$prediction <-
            cbind(crossValidationResults$prediction, value = NA)
          attr(
            crossValidationResults$prediction,
            "metaData"
          )$modelType <- "binary"
          crossValidationResults$param <- parameters
          crossValidationResults$param$learnSchedule <- list(
            LRs = NA,
            bestEpoch = NA
          )
          nFolds <- max(labels$index)
          hyperSummary <- data.frame(
            metric = rep(tuningMetric$name, nFolds + 1),
            fold = c("CV", as.character(1:nFolds)),
            value = NA
          )
          hyperSummary <- cbind(hyperSummary, parameters)
          hyperSummary$learnRates <- NA

          gridPerformance <- list(
            metric = tuningMetric$name,
            cvPerformance = NA,
            cvPerformancePerFold = rep(NA, nFolds),
            param = parameters,
            hyperSummary = hyperSummary
          )
          crossValidationResults$gridPerformance <- gridPerformance
          learnRates <- list()
          for (i in 1:nFolds) {
            learnRates[[i]] <- list(
              LRs = NA,
              bestEpoch = NA
            )
          }
          crossValidationResults$learnRates <- learnRates
          return(crossValidationResults)
        } else {
          stop(e)
        }
      }
    )
  gridSearchPredictions <- list(
    prediction = crossValidationResults$prediction,
    param = parameters,
    gridPerformance = crossValidationResults$gridPerformance
  )
  maxIndex <- which.max(unlist(sapply(
    crossValidationResults$learnRates,
    `[`,
    2
  )))
  if (length(maxIndex) != 0) {
    gridSearchPredictions$gridPerformance$hyperSummary$learnRates <-
      rep(
        list(unlist(crossValidationResults$learnRates[[maxIndex]]$LRs)),
        nrow(gridSearchPredictions$gridPerformance$hyperSummary)
      )
    gridSearchPredictions$param$learnSchedule <-
      crossValidationResults$learnRates[[maxIndex]]
  }
  return(gridSearchPredictions)
}

doCrossValidationImpl <- function(
    dataset,
    labels,
    parameters,
    modelSettings,
    tuningMetric) {
  fitParams <- names(parameters)[grepl(
    "^estimator",
    names(parameters)
  )]
  modelParamNames <- setdiff(
    names(parameters)[!grepl("^estimator", names(parameters))],
    "learnSchedule"
  )
  currentModelParams <- parameters[modelParamNames]
  attr(currentModelParams, "metaData")$names <- modelParamNames
  currentModelParams$modelType <- modelSettings$modelType
  currentEstimatorSettings <- fillEstimatorSettings(
    modelSettings$estimatorSettings,
    fitParams,
    parameters
  )
  currentModelParams$feature_info <- dataset$get_feature_info()
  currentParameters <- list(
    modelParameters = currentModelParams,
    estimatorSettings = currentEstimatorSettings
  )
  if (currentEstimatorSettings$findLR) {
    lr <- getLR(currentParameters, dataset)
    ParallelLogger::logInfo(paste0("Auto learning rate selected as: ", lr))
    currentParameters$estimatorSettings$learningRate <- lr
    currentParameters$estimatorSettings$findLR <- FALSE
  }

  fold <- labels$index
  ParallelLogger::logInfo(paste0("Max fold: ", max(fold)))
  if (isTRUE(modelSettings$estimatorSettings$trainValidationSplit)) {
    ParallelLogger::logInfo("Using train-validation split instead of cross validation:
      training on folds != max(folds), and validating on fold == max(folds)")
    fold <- ifelse(fold == max(fold), 1, 0)
    labels$index <- fold
  }
  learnRates <- list()
  prediction <- NULL
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  fit_estimator <- reticulate::import_from_path(
    "Estimator",
    path = path
  )$fit_estimator
  for (i in 1:max(fold)) {
    ParallelLogger::logInfo(paste0("Fold ", i))

    # -1 for python 0-based indexing
    trainDataset <- torch$utils$data$Subset(
      dataset,
      indices = as.integer(which(fold != i) - 1)
    )

    # -1 for python 0-based indexing
    testDataset <- torch$utils$data$Subset(
      dataset,
      indices = as.integer(which(fold == i) - 1)
    )
    estimator <- createEstimator(currentParameters)
    fit_estimator(estimator, trainDataset, testDataset)

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
  gridPerformance <- computeDeepGridPerformance(
    prediction,
    parameters,
    tuningMetric
  )
  return(
    results = list(
      prediction = prediction,
      learnRates = learnRates,
      gridPerformance = gridPerformance
    )
  )
}

computeDeepGridPerformance <- function(prediction, parameters, tuningMetric) {
  if (is.null(prediction$index)) {
    cvPerformance <- tuningMetric$fun(prediction)
    hyperSummary <- data.frame(
      metric = tuningMetric$name,
      fold = "CV",
      value = cvPerformance
    )
    hyperSummary <- cbind(hyperSummary, parameters)
    return(list(
      metric = tuningMetric$name,
      cvPerformance = cvPerformance,
      cvPerformancePerFold = cvPerformance,
      param = parameters,
      hyperSummary = hyperSummary
    ))
  }

  folds <- sort(unique(prediction$index))
  cvPerformancePerFold <- vapply(
    folds,
    function(fold) tuningMetric$fun(prediction[prediction$index == fold, ]),
    numeric(1)
  )
  cvPerformance <- mean(cvPerformancePerFold, na.rm = TRUE)
  hyperSummary <- data.frame(
    metric = tuningMetric$name,
    fold = c("CV", as.character(folds)),
    value = c(cvPerformance, cvPerformancePerFold)
  )
  hyperSummary <- cbind(hyperSummary, parameters)
  list(
    metric = tuningMetric$name,
    cvPerformance = cvPerformance,
    cvPerformancePerFold = cvPerformancePerFold,
    param = parameters,
    hyperSummary = hyperSummary
  )
}


extractParamsToTune <- function(estimatorSettings) {
  paramsToTune <- list()
  for (name in names(estimatorSettings)) {
    param <- estimatorSettings[[name]]
    if (length(param) > 1 && is.atomic(param)) {
      paramsToTune[[paste0("estimator.", name)]] <- param
    }
    if ("params" %in% names(param)) {
      for (name2 in names(param[["params"]])) {
        param2 <- param[["params"]][[name2]]
        if (length(param2) > 1) {
          paramsToTune[[paste0("estimator.", name, ".", name2)]] <- param2
        }
      }
    }
  }
  return(paramsToTune)
}

trainFinalModel <- function(dataset, finalParam, modelSettings, labels) {
  # get the params
  modelParamNames <- setdiff(
    names(finalParam)[!grepl("^estimator", names(finalParam))],
    "learnSchedule"
  )
  modelParams <- finalParam[modelParamNames]

  fitParams <- names(finalParam)[grepl("^estimator", names(finalParam))]

  modelParams$featureInfo <- dataset$get_feature_info()
  modelParams$modelType <- modelSettings$modelType

  estimatorSettings <- fillEstimatorSettings(
    modelSettings$estimatorSettings,
    fitParams,
    finalParam
  )
  if (!startsWith(modelSettings$modelType, "RealMLP")) {
    estimatorSettings$learningRate <- finalParam$learnSchedule$LRs[[1]]
  }
  parameters <- list(
    modelParameters = modelParams,
    estimatorSettings = estimatorSettings
  )
  estimator <- createEstimator(parameters = parameters)
  estimator$fit_whole_training_set(dataset, finalParam$learnSchedule$LRs)

  ParallelLogger::logInfo("Calculating predictions on all train data...")
  prediction <- predictDeepEstimator(
    plpModel = estimator,
    data = dataset,
    cohort = labels
  )
  prediction$evaluationType <- "Train"
  return(list(prediction = prediction, estimator = estimator))
}
