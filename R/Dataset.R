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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
createDataset <- function(data, labels, 
                          plpModel = NULL, 
                          maxSequenceLength = NULL,
                          truncation = NULL) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  attributes(data)$path <- attributes(data)$path %||% attributes(data)$dbname
  
  args <- list(
    r_to_py(normalizePath(attributes(data)$path))
  )
  
  if (!"timeId" %in% names(data$covariates)) { 
    dataset <- reticulate::import_from_path("Dataset", path = path)$Data
  } else {
    dataset <- reticulate::import_from_path("Dataset", path = path)$TemporalData
    args$max_sequence_length <- r_to_py(maxSequenceLength)
    args$truncation <- r_to_py(truncation)
  }
  
  # training
  if (is.null(plpModel)) {
    args$labels <- r_to_py(labels$outcomeCount)
    return(do.call(dataset, args))
  }

  # backwards compatibility with numerical index either from data from plpModel
  numericalIndex <- data$numericalIndex %||% plpModel$covariateImportance$isNumeric
  if (!is.null(numericalIndex)) {
    args$numericalIndex <- r_to_py(as.array(numericalIndex %>% dplyr::pull()))
    return(do.call(dataset, args))
  }

  # testing
  args$data_reference <- r_to_py(plpModel$covariateImportance)
  return(do.call(dataset, args))
}
