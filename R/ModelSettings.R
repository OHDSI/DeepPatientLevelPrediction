# @file ModelSettings.R
#
# Copyright 2026 Observational Health Data Sciences and Informatics
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

# Builds the modelSettings of every model: the hyperparameter combinations of
# paramGrid and of the estimator settings with several values, sampled when
# hyperParamSearch is "random".
#
# prepareCandidate, if given, is applied to each combination, for settings that
# depend on others, such as a hidden size given as a ratio of the token size.
createDeepModelSettings <- function(modelType,
                                    paramGrid,
                                    estimatorSettings,
                                    hyperParamSearch,
                                    randomSample,
                                    randomSampleSeed,
                                    prepareCandidate = NULL) {
  checkIsClass(hyperParamSearch, "character")
  checkInStringVector(hyperParamSearch, c("random", "grid"))

  checkIsClass(randomSample, c("numeric", "integer"))
  checkHigherEqual(randomSample, 1)

  checkIsClass(randomSampleSeed, c("numeric", "integer", "NULL"))

  param <- PatientLevelPrediction::listCartesian(
    c(paramGrid, estimatorSettings$paramsToTune)
  )
  if (!is.null(prepareCandidate)) {
    param <- lapply(param, prepareCandidate)
  }

  if (hyperParamSearch == "random") {
    # asking for more samples than there are combinations runs each once, so
    # that the default randomSample works with one combination
    randomSample <- min(randomSample, length(param))
    suppressWarnings(withr::with_seed(randomSampleSeed, {
      param <- param[sample(
        length(param),
        randomSample
      )]
    }))
  }

  results <- list(
    fitFunction = "DeepPatientLevelPrediction::fitEstimator",
    param = param,
    estimatorSettings = estimatorSettings,
    saveType = "file",
    modelType = modelType
  )
  attr(results$param, "settings")$modelType <- modelType
  class(results) <- "modelSettings"
  results
}
