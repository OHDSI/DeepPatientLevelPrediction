# @file Dataset.R
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
createDataset <- function(data, labels, plpModel = NULL) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  dataset <- reticulate::import_from_path("Dataset", path = path)$Data
  if (is.null(attributes(data)$path)) {
    # sqlite object
    attributes(data)$path <- attributes(data)$dbname
  }
  if (is.null(plpModel) && is.null(data$numericalIndex)) {
    data <- dataset(r_to_py(normalizePath(attributes(data)$path)),
                    r_to_py(labels$outcomeCount))
  } 
  else if (!is.null(data$numericalIndex)) {
    numericalIndex <- 
      r_to_py(as.array(data$numericalIndex %>% dplyr::pull()))
    data <- dataset(r_to_py(normalizePath(attributes(data)$path)),
                    r_to_py(labels$outcomeCount),
                    numericalIndex) 
  }
  else {
    numericalFeatures <-
      r_to_py(as.array(which(plpModel$covariateImportance$isNumeric)))
    data <- dataset(r_to_py(normalizePath(attributes(data)$path)),
                    numerical_features = numericalFeatures)
  }

  return(data)
}
