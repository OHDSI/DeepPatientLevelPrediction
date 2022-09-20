# @file SaveLoad.R
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

#' Load estimator model from disk
#'
#' @description
#' Load estimator model from disk.
#'
#' @param path Path to model directory.
#' @param device Provide device. Can be \code{'cpu'} or \code{'cuda'}
#' 
#' @export
loadEstimator <- function(path, device = 'cpu') {
  
  if (!dir.exists(path)) {
    stop(paste0("Directory does not exist at ", path, "."))
  }
  
  if (!file.exists(file.path(path, "model", "DeepEstimatorModel.pt"))) {
    stop(paste0("Model does not exist at ", path, "."))
  }

  model <- torch::torch_load(file.path(path, "model", "DeepEstimatorModel.pt"), device = "cpu")
  estimator <- Estimator$new(
    baseModel = model$fitParameters$baseModel,
    modelParameters = model$modelParameters,
    fitParameters = model$fitParameters,
    device = device
  )
  estimator$model$load_state_dict(model$modelStateDict)
  
  return(estimator) 
}



