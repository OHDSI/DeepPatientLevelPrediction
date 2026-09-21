# @file TransferLearning.R
#
# Copyright 2023 Observational Health Data Sciences and Informatics
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

#' Create Fine-Tuning Settings
#'
#' Creates settings for fine-tuning a previously fitted deep learning model.
#'
#' @name setFinetuner
#' @param modelPath Path to an existing saved `plpModel` directory.
#' @param estimatorSettings Estimator settings created by [setEstimator()].
#'
#' @return A `modelSettings` object that initializes from the saved model.
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
#'
#' # Save a real fitted model before creating settings for a new training task.
#' modelPath <- file.path(analysisPath, "savedModel")
#' PatientLevelPrediction::savePlpModel(model, modelPath)
#' finetuneSettings <- setFinetuner(
#'   modelPath = modelPath,
#'   estimatorSettings = setEstimator(
#'     learningRate = 0.001, batchSize = 64, epochs = 1, seed = 42
#'   )
#' )
#' finetuneSettings$modelType
#'
#' # Clean up this example's files. In real use, keep modelPath until
#' # fine-tuning finishes: finetuneSettings refers to the saved model files.
#' Andromeda::close(splitData$Train$covariateData)
#' Andromeda::close(plpData$covariateData)
#' unlink(analysisPath, recursive = TRUE)
#' unlink(model$model, recursive = TRUE)
#' }
#' @export
setFinetuner <- function(modelPath,
                         estimatorSettings = setEstimator()) {

  if (!dir.exists(modelPath)) {
    stop(paste0("supplied modelPath does not exist, you supplied: modelPath = ",
                modelPath))
  }
  # TODO check if it's a valid path to a plpModel
  if (!dir.exists(file.path(modelPath, "model"))) {
    stop(paste0("supplied modelPath does not contain a model directory, you supplied: modelPath = ",
                modelPath))
  }
  if (!file.exists(file.path(modelPath, "model", "DeepEstimatorModel.pt"))) {
    stop(paste0("supplied modelPath does not contain a model file, you supplied: modelPath = ",
                modelPath))
  }
  
  plpModel <- PatientLevelPrediction::loadPlpModel(modelPath)
  estimatorSettings$finetuneModelPath <-
    normalizePath(file.path(plpModel$model, "DeepEstimatorModel.pt"))
  modelType <-
    plpModel$modelDesign$modelSettings$modelType
  
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  estimatorSettings$initStrategy <-
    reticulate::import_from_path("InitStrategy",
                                 path = path)$FinetuneInitStrategy()
  
  param <- list()
  param[[1]] <- list(modelPath = modelPath)

  results <- list(
    fitFunction = "fitEstimator",
    param = param,
    estimatorSettings = estimatorSettings,
    saveType = "file",
    modelParamNames = c("modelPath"),
    modelType = modelType
  )
  attr(results$param, "settings")$modelType <- "Finetuner"

  class(results) <- "modelSettings"

  return(results)
}
