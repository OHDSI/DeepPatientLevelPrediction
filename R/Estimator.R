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

#' Create Estimator Settings
#'
#' Creates settings controlling optimization, model fitting, and model
#' selection for a deep learning estimator.
#'
#' @name setEstimator
#' @param learningRate Learning rate, or `"auto"` to use the learning-rate
#'   finder.
#' @param weightDecay Weight-decay value.
#' @param batchSize Number of observations per batch.
#' @param epochs Maximum number of training epochs.
#' @param device Device on which to train. This can be a string or a function
#'   evaluated when training begins.
#' @param optimizer PyTorch optimizer constructor. Evaluation is delayed until
#'   training begins.
#' @param scheduler A list containing the learning-rate scheduler constructor
#'   in `fun` and its arguments in `params`. Evaluation is delayed until
#'   training begins.
#' @param criterion PyTorch loss constructor. Evaluation is delayed until
#'   training begins.
#' @param earlyStopping Early-stopping settings, or `NULL` to disable early
#'   stopping.
#' @param compile Whether to compile the PyTorch model before training.
#' @param metric Either `"auc"`, `"loss"`, or a list defining a custom metric.
#'   A custom metric list must contain `fun`, `mode`, and `name`.
#' @param accumulationSteps Number of batches over which to accumulate
#'   gradients, or a function evaluated when training begins.
#' @param seed Random seed used to initialize the model. A seed is generated
#'   when this is `NULL`.
#' @param trainValidationSplit Whether to use a train-validation split for
#'   model selection instead of cross-validation.
#' @param initOutputBias Whether to start the bias of the output layer at the
#'   log-odds of the outcome rate in the training data, so that the first
#'   predictions match the outcome rate instead of 50%. Without it, models for
#'   rare outcomes spend their first epochs lowering all predictions, and early
#'   stopping can end training before they learn to rank patients. It applies to
#'   models whose output is a single logit from a linear `head` layer and is
#'   not used during the learning-rate search.
#'
#' @return A list of estimator settings used by the model-setting functions.
#'
#' @examples
#' estimatorSettings <- setEstimator(
#'   learningRate = 0.001,
#'   batchSize = 128,
#'   epochs = 10,
#'   seed = 42
#' )
#' estimatorSettings$batchSize
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
    trainValidationSplit = FALSE,
    initOutputBias = TRUE) {
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
  checkIsClass(initOutputBias, "logical")
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
  generatedSeed <- NULL
  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
    generatedSeed <- seed
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
    seed = seed[1],
    initOutputBias = initOutputBias
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
  attr(estimatorSettings, "generatedSeed") <- generatedSeed
  return(estimatorSettings)
}

#' Fit a Deep Learning Estimator
#'
#' Fits a configured deep learning model to a prepared
#' `PatientLevelPrediction` training-data object.
#'
#' @param trainData A prepared training-data object produced by
#'   `PatientLevelPrediction`.
#' @param modelSettings A `modelSettings` object created by one of this
#'   package's model-setting functions.
#' @param analysisId Identifier for the analysis.
#' @param analysisPath Directory used for training artifacts and the resumable
#'   training cache.
#' @param hyperparameterSettings Hyperparameter settings from
#'   `PatientLevelPrediction::createHyperparameterSettings()`. Its tuning metric
#'   selects the best hyperparameter combination. Its search settings are
#'   ignored with a warning, because the search is defined in the model
#'   settings.
#' @param ... Additional arguments reserved for the
#'   `PatientLevelPrediction` model interface.
#'
#' @return A `plpModel` object containing the fitted model, predictions,
#'   training details, and covariate information.
#'
#' @examplesIf rlang::is_installed("FeatureExtraction")
#' \dontrun{
#' # Requires FeatureExtraction and the package's Python dependencies,
#' # including PyTorch.
#' # See vignette("Installing") for setup instructions.
#' data("simulationProfile", package = "PatientLevelPrediction")
#' plpData <- PatientLevelPrediction::simulatePlpData(
#'   simulationProfile, n = 200, seed = 42
#' )
#' # Supply metadata omitted by simulatePlpData() for this bundled profile:
#' # gender, age, conditions, and drugs. Only age is continuous.
#' plpData$covariateData$analysisRef <- data.frame(
#'   analysisId = c(1, 2, 102, 402),
#'   isBinary = c("Y", "N", "Y", "Y"),
#'   missingMeansZero = c(NA, "Y", NA, NA)
#' )
#' population <- PatientLevelPrediction::createStudyPopulation(
#'   plpData,
#'   populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
#'     riskWindowEnd = 90, minTimeAtRisk = 89
#'   )
#' )
#' splitData <- PatientLevelPrediction::splitData(
#'   plpData,
#'   population,
#'   splitSettings = PatientLevelPrediction::createDefaultSplitSetting(
#'     testFraction = 0, trainFraction = 1, nfold = 2, splitSeed = 42
#'   )
#' )
#'
#' # Keep this toy example small: one configuration, two folds, and one epoch.
#' # These settings demonstrate fitting, not meaningful predictive performance.
#' modelSettings <- setResNet(
#'   numLayers = 1,
#'   sizeHidden = 8,
#'   hiddenFactor = 1,
#'   residualDropout = 0,
#'   hiddenDropout = 0,
#'   sizeEmbedding = 8,
#'   hyperParamSearch = "grid",
#'   estimatorSettings = setEstimator(
#'     learningRate = 0.001, batchSize = 64, epochs = 1, seed = 42
#'   )
#' )
#' analysisPath <- tempfile("deep-plp-example-")
#' dir.create(analysisPath)
#' model <- fitEstimator(
#'   trainData = splitData$Train,
#'   modelSettings = modelSettings,
#'   analysisId = 1,
#'   analysisPath = analysisPath
#' )
#' head(model$prediction)
#'
#' # Clean up this example's data, cache, and fitted-model files.
#' # In real use, retain model$model until you have saved or finished using it.
#' Andromeda::close(splitData$Train$covariateData)
#' Andromeda::close(plpData$covariateData)
#' unlink(analysisPath, recursive = TRUE)
#' unlink(model$model, recursive = TRUE)
#' }
#'
#' @export
fitEstimator <- function(
    trainData,
    modelSettings,
    analysisId,
    analysisPath,
    hyperparameterSettings = NULL,
    ...) {
  start <- Sys.time()
  warnIgnoredSearchSettings(hyperparameterSettings)
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
      modelLocation = outLoc,
      analysisPath = analysisPath,
      tuningMetric = hyperparameterSettings$tuningMetric
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
    )

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
      requiresDenseMatrix = FALSE
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
      splitSettings = attr(trainData, "metaData")$splitSettings,
      sampleSettings = attr(trainData, "metaData")$sampleSettings
    ),
    trainDetails = list(
      analysisId = analysisId,
      analysisSource = "",
      developmentDatabase = attr(trainData, "metaData")$cdmDatabaseName,
      developmentDatabaseSchema = attr(trainData, "metaData")$cdmDatabaseSchema,
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

#' Predict with a Deep Learning Estimator
#'
#' Applies a fitted deep learning model to prepared patient-level prediction
#' data.
#'
#' @param plpModel A fitted `plpModel` or an in-memory Python estimator.
#' @param data A `plpData` object or a compatible PyTorch dataset.
#' @param cohort A data frame containing the row identifiers to score.
#'
#' @return A data frame containing the input cohort columns and predicted
#'   values in the `value` column.
#'
#' @examplesIf rlang::is_installed("FeatureExtraction")
#' \dontrun{
#' # Requires FeatureExtraction and the package's Python dependencies,
#' # including PyTorch.
#' # See vignette("Installing") for setup instructions.
#' data("simulationProfile", package = "PatientLevelPrediction")
#' plpData <- PatientLevelPrediction::simulatePlpData(
#'   simulationProfile, n = 200, seed = 42
#' )
#' # Supply metadata omitted by simulatePlpData() for this bundled profile:
#' # gender, age, conditions, and drugs. Only age is continuous.
#' plpData$covariateData$analysisRef <- data.frame(
#'   analysisId = c(1, 2, 102, 402),
#'   isBinary = c("Y", "N", "Y", "Y"),
#'   missingMeansZero = c(NA, "Y", NA, NA)
#' )
#' population <- PatientLevelPrediction::createStudyPopulation(
#'   plpData,
#'   populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
#'     riskWindowEnd = 90, minTimeAtRisk = 89
#'   )
#' )
#' splitData <- PatientLevelPrediction::splitData(
#'   plpData,
#'   population,
#'   splitSettings = PatientLevelPrediction::createDefaultSplitSetting(
#'     testFraction = 0.25, trainFraction = 0.75, nfold = 2, splitSeed = 42
#'   )
#' )
#'
#' # Keep this toy example small: one configuration, two folds, and one epoch.
#' # These settings demonstrate fitting, not meaningful predictive performance.
#' modelSettings <- setResNet(
#'   numLayers = 1,
#'   sizeHidden = 8,
#'   hiddenFactor = 1,
#'   residualDropout = 0,
#'   hiddenDropout = 0,
#'   sizeEmbedding = 8,
#'   hyperParamSearch = "grid",
#'   estimatorSettings = setEstimator(
#'     learningRate = 0.001, batchSize = 64, epochs = 1, seed = 42
#'   )
#' )
#' analysisPath <- tempfile("deep-plp-example-")
#' dir.create(analysisPath)
#' model <- fitEstimator(
#'   trainData = splitData$Train,
#'   modelSettings = modelSettings,
#'   analysisId = 1,
#'   analysisPath = analysisPath
#' )
#'
#' # Score held-out test patients using only the fitted training model.
#' prediction <- predictDeepEstimator(
#'   plpModel = model,
#'   data = splitData$Test,
#'   cohort = splitData$Test$labels
#' )
#' head(prediction)
#'
#' # Clean up after inspecting predictions.
#' Andromeda::close(splitData$Test$covariateData)
#' Andromeda::close(splitData$Train$covariateData)
#' Andromeda::close(plpData$covariateData)
#' unlink(analysisPath, recursive = TRUE)
#' unlink(model$model, recursive = TRUE)
#' }
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
      temporalSettings = attr(plpModel$modelDesign$modelSettings$param, "temporalSettings"),
      numRows = nrow(cohort)
    )
  } else if ("plpData" %in% class(data)) {
    mappedData <- PatientLevelPrediction::MapIds(
      data$covariateData,
      cohort = cohort,
      mapping = plpModel$covariateImportance %>%
        dplyr::select("columnId", "covariateId")
    )
    data <- createDataset(mappedData,
      plpModel = plpModel,
      numRows = nrow(cohort)
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
    predictions <- estimator$predict_proba(data)
  } else {
    predictions <- plpModel$model$predict_proba(data)
  }
  predictions <- as.numeric(predictions)
  if (length(predictions) != nrow(prediction)) {
    stop(paste0(
      "Got ", length(predictions), " predictions for ", nrow(prediction),
      " patients in the cohort"
    ))
  }
  prediction$value <- predictions
  attr(prediction, "metaData")$modelType <- attr(plpModel, "modelType")

  return(prediction)
}

#' Tune a Deep Learning Estimator
#'
#' Performs hyperparameter search and cross-validation, then fits a final model
#' using the best-performing settings.
#'
#' @param mappedData Covariate data mapped with
#'   `PatientLevelPrediction::MapIds()`.
#' @param labels Data frame containing outcomes and fold assignments.
#' @param modelSettings A `modelSettings` object.
#' @param modelLocation Directory in which to save the fitted model.
#' @param analysisPath Directory used for the resumable training cache.
#' @param tuningMetric A tuning metric from
#'   `PatientLevelPrediction::createTuningMetric()` used to select the best
#'   hyperparameter combination. The default `NULL` uses the AUC.
#'
#' @return A list containing the saved estimator location, predictions, final
#'   parameters, hyperparameter-search summaries, and feature information.
#'
#' @examplesIf rlang::is_installed("FeatureExtraction")
#' \dontrun{
#' # Requires FeatureExtraction and the package's Python dependencies,
#' # including PyTorch.
#' # See vignette("Installing") for setup instructions.
#' data("simulationProfile", package = "PatientLevelPrediction")
#' plpData <- PatientLevelPrediction::simulatePlpData(
#'   simulationProfile, n = 200, seed = 42
#' )
#' # Supply metadata omitted by simulatePlpData() for this bundled profile:
#' # gender, age, conditions, and drugs. Only age is continuous.
#' plpData$covariateData$analysisRef <- data.frame(
#'   analysisId = c(1, 2, 102, 402),
#'   isBinary = c("Y", "N", "Y", "Y"),
#'   missingMeansZero = c(NA, "Y", NA, NA)
#' )
#' population <- PatientLevelPrediction::createStudyPopulation(
#'   plpData,
#'   populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
#'     riskWindowEnd = 90, minTimeAtRisk = 89
#'   )
#' )
#' splitData <- PatientLevelPrediction::splitData(
#'   plpData,
#'   population,
#'   splitSettings = PatientLevelPrediction::createDefaultSplitSetting(
#'     testFraction = 0, trainFraction = 1, nfold = 2, splitSeed = 42
#'   )
#' )
#'
#' # Keep this toy example small: two configurations, two folds, and one epoch.
#' # These settings demonstrate fitting, not meaningful predictive performance.
#' modelSettings <- setResNet(
#'   numLayers = 1,
#'   sizeHidden = c(8, 16),
#'   hiddenFactor = 1,
#'   residualDropout = 0,
#'   hiddenDropout = 0,
#'   sizeEmbedding = 8,
#'   hyperParamSearch = "grid",
#'   estimatorSettings = setEstimator(
#'     learningRate = 0.001, batchSize = 64, epochs = 1, seed = 42
#'   )
#' )
#' analysisPath <- tempfile("deep-plp-example-")
#' dir.create(analysisPath)
#'
#' # The lower-level interface needs mapped IDs and fold assignments in labels.
#' labels <- merge(splitData$Train$labels, splitData$Train$folds, by = "rowId")
#' mappedData <- PatientLevelPrediction::MapIds(
#'   covariateData = splitData$Train$covariateData,
#'   cohort = labels
#' )
#' result <- gridCvDeep(
#'   mappedData = mappedData,
#'   labels = labels,
#'   modelSettings = modelSettings,
#'   modelLocation = file.path(analysisPath, "model"),
#'   analysisPath = analysisPath
#' )
#' result$finalParam
#'
#' # Clean up this example's data, cache, and final model.
#' Andromeda::close(mappedData)
#' Andromeda::close(splitData$Train$covariateData)
#' Andromeda::close(plpData$covariateData)
#' unlink(analysisPath, recursive = TRUE)
#' }
#'
#' @export
gridCvDeep <- function(
    mappedData,
    labels,
    modelSettings,
    modelLocation,
    analysisPath,
    tuningMetric = NULL) {
  ParallelLogger::logInfo(paste0(
    "Running hyperparameter search for ",
    modelSettings$modelType,
    " model"
  ))
  ###########################################################################

  paramSearch <- modelSettings$param

  dataset <- createDataset(
    data = mappedData,
    labels = labels,
    temporalSettings = attr(paramSearch, "temporalSettings")
  )
  maximize <- tuningMetric$maximize %||% TRUE

  # setup cache for hyperparameterResults
  trainCache <- setupCache(
    analysisPath,
    paramSearch,
    createCacheFingerprint(dataset, labels, modelSettings, tuningMetric)
  )
  hyperparameterResults <- trainCache$getGridSearchPredictions()

  if (!trainCache$isFull()) {
    for (gridId in trainCache$getLastGridSearchIndex():length(paramSearch)) {
      ParallelLogger::logInfo(paste0(
        "Running hyperparameter combination no ",
        gridId
      ))
      ParallelLogger::logInfo(paste0("HyperParameters: "))
      ParallelLogger::logInfo(paste(
        names(paramSearch[[gridId]]),
        paramSearch[[gridId]],
        collapse = " | "
      ))
      hyperparameterResults[[gridId]] <-
        doCrossValidation(
          dataset,
          labels = labels,
          parameters = paramSearch[[gridId]],
          modelSettings = modelSettings,
          tuningMetric = tuningMetric
        )
      # remove all predictions that are not the best performance
      hyperparameterResults <- trainCache$trimPerformance(
        hyperparameterResults,
        maximize = maximize
      )
      trainCache$saveGridSearchPredictions(hyperparameterResults)
    }
  }
  paramGridSearch <- lapply(
    hyperparameterResults,
    function(x) x$gridPerformance
  )
  # get best params
  indexOfBest <- getBestGridIndex(hyperparameterResults, maximize)
  if (length(indexOfBest) == 0) {
    stop("No hyperparameter combination has valid results")
  }
  finalParam <- hyperparameterResults[[indexOfBest]]$param

  paramGridSearch <- lapply(
    hyperparameterResults,
    function(x) x$gridPerformance
  )
  # get best CV prediction
  cvPrediction <- hyperparameterResults[[indexOfBest]]$prediction
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
      featureInfo = featureInfo
    )
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

doCrossValidation <- function(
    dataset,
    labels,
    parameters,
    modelSettings,
    tuningMetric = NULL) {
  metricName <- tuningMetric$name %||% "computeAuc"
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
            metric = rep(metricName, nFolds + 1),
            fold = c("CV", as.character(1:nFolds)),
            value = NA
          )
          hyperSummary <- cbind(hyperSummary, parameters)
          hyperSummary$learnRates <- NA

          gridPerformance <- list(
            metric = metricName,
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
    tuningMetric = NULL) {
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
    foldParameters <- currentParameters
    if (currentEstimatorSettings$findLR) {
      # search on the training part only so the held-out outcomes stay unused
      lr <- getLR(foldParameters, trainDataset)
      ParallelLogger::logInfo(paste0("Auto learning rate selected as: ", lr))
      foldParameters$estimatorSettings$learningRate <- lr
      foldParameters$estimatorSettings$findLR <- FALSE
    }
    estimator <- createEstimator(foldParameters)
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
  gridPerformance <- computeTuningPerformance(
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


computeTuningPerformance <- function(prediction, parameters, tuningMetric = NULL) {
  if (is.null(tuningMetric)) {
    return(PatientLevelPrediction::computeGridPerformance(prediction, parameters))
  }
  gridPerformance <- PatientLevelPrediction::computeGridPerformance(
    prediction,
    parameters,
    performanceFunct = tuningMetric$fun
  )
  gridPerformance$metric <- tuningMetric$name
  gridPerformance$hyperSummary$metric <- tuningMetric$name
  return(gridPerformance)
}

# position of the best combination, ignoring combinations without results
getBestGridIndex <- function(hyperparameterResults, maximize = TRUE) {
  performance <- vapply(
    hyperparameterResults,
    function(x) as.numeric(x$gridPerformance$cvPerformance %||% NA_real_),
    numeric(1)
  )
  if (maximize) {
    return(which.max(performance))
  }
  return(which.min(performance))
}

# Only the tuning metric of PatientLevelPrediction's hyperparameter settings is used.
warnIgnoredSearchSettings <- function(hyperparameterSettings) {
  if (is.null(hyperparameterSettings)) {
    return(invisible(NULL))
  }
  defaults <- PatientLevelPrediction::createHyperparameterSettings()
  searchSettings <- c("search", "sampleSize", "randomSeed", "generator")
  ignored <- searchSettings[!vapply(
    searchSettings,
    function(name) identical(hyperparameterSettings[[name]], defaults[[name]]),
    logical(1)
  )]
  if (length(ignored) > 0) {
    warning(
      "DeepPatientLevelPrediction ignores ",
      paste0("`", ignored, "`", collapse = ", "),
      " in hyperparameterSettings and uses only its tuning metric. Set the ",
      "search with `hyperParamSearch` and `randomSample` in the model settings, ",
      "for example in setResNet().",
      call. = FALSE
    )
  }
  return(invisible(ignored))
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
