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
                          temporalSettings = NULL,
                          numRows = NULL) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")

  newPath <- tempfile()
  Andromeda::saveAndromeda(data, newPath, maintainConnection = TRUE)

  attributes(data)$path <- attributes(data)$path %||% attributes(data)$dbname

  args <- list(
    r_to_py(normalizePath(newPath))
  )

  dataset <- reticulate::import_from_path("Dataset", path = path)$Data
  if ("timeId" %in% names(data$covariates)) {
    args$temporal_settings <- r_to_py(camelCaseToSnakeCaseNames(temporalSettings))
  }
  # training
  if (is.null(plpModel)) {
    args$labels <- r_to_py(labels$outcomeCount)
    return(do.call(dataset, args))
  }

  # testing
  args$data_reference <- rToPolars(plpModel$covariateImportance)
  if (!is.null(numRows)) {
    args$num_rows <- as.integer(numRows)
  }
  return(do.call(dataset, args))
}
