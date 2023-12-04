# @file LRFinder.R
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
createLRFinder <- function(modelType,
                           modelParameters,
                           estimatorSettings,
                           lrSettings = NULL) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  lrFinderClass <-
    reticulate::import_from_path("LrFinder", path = path)$LrFinder
  
  

  model <- reticulate::import_from_path(modelType, path = path)[[modelType]]
  modelParameters <- camelCaseToSnakeCaseNames(modelParameters)
  estimatorSettings <- camelCaseToSnakeCaseNames(estimatorSettings)
  estimatorSettings <- evalEstimatorSettings(estimatorSettings)
  
  # estimator <- createEstimator(modelType = estimatorSettings$modelType,
  #                              modelParameters = modelParameters,
  #                              estimatorSettings = estimatorSettings)
  if (!is.null(lrSettings)) {
    lrSettings <- camelCaseToSnakeCaseNames(lrSettings)
  }
  
  lrFinder <- lrFinderClass(model = model,
                            model_parameters = modelParameters,
                            estimator_settings = estimatorSettings,
                            lr_settings = lrSettings)

  return(lrFinder)
}
