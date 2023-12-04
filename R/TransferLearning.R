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

#' setFinetuner
#'
#' @description
#' creates settings for using transfer learning to finetune a model
#'
#' @name setFinetuner
#' @param modelPath path to existing plpModel directory
#' @param estimatorSettings settings created with `setEstimator`
#' @export
setFinetuner <- function(modelPath,
                         estimatorSettings =
                           setEstimator(learningRate = learningRate,
                                        weightDecay = weightDecay,
                                        batchSize = batchSize,
                                        epochs = epochs,
                                        device = device,
                                        optimizer = optimizer,
                                        scheduler = scheduler,
                                        criterion = criterion,
                                        earlyStopping = earlyStopping,
                                        metric = metric,
                                        seed = seed)
)  {

  if (!dir.exists(modelPath)) {
    stop(paste0("supplied modelPath does not exist, you supplied: modelPath = ", 
                modelPath))
  }
  param <- list()
  param[[1]] <- list(modelPath = modelPath)
    
  attr(param, "settings")$modelType <- "FineTuner"
  results <- list(
  fitFunction = "fitEstimator",
  param = param,
  estimatorSettings = estimatorSettings,
  modelType = "Finetuner",
  saveType = "file",
  modelParamNames = c("modelPath")
  )
 
 class(results) <- "modelSettings"
 
 return(results)
}