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
#' @examples
#' \dontrun{
#' # Requires a previously saved DeepPatientLevelPrediction model directory.
#' finetuneSettings <- setFinetuner(
#'   modelPath = "path/to/savedPlpModel",
#'   estimatorSettings = setEstimator(epochs = 5)
#' )
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
