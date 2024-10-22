# @file CustomEmbeddingModel.R
#
# Copyright 2024 Observational Health Data Sciences and Informatics
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
#' Create default settings a model using custom embeddings
#'
#' @description A model that uses custom embeddings such as Poincare embeddings or 
#' embeddings form a foundation model
#' @param embeddingFilePath path to the saved embeddings. The embeddings file 
#' should be a pytorch file including a dictionary with two two fields: 
#' `concept_ids`: a pytorch long tensor with the concept ids and `embeddings`: 
#' a pytorch float tensor with the embeddings
#' @param modelSettings for the model to use, needs to have an embedding layer 
#' with a name `embedding` which will be replaced by the custom embeddings
#' 
#' @return settings for a model using custom embeddings
#'
#' @export
setCustomEmbeddingModel <- function(
    embeddingFilePath,
    modelSettings = setTransformer(
      numBlocks = 3,
      dimToken = 16,
      dimOut = 1,
      numHeads = 4,
      attDropout = 0.2,
      ffnDropout = 0.1,
      resDropout = 0.0,
      dimHidden = 32,
      estimatorSettings = setEstimator(learningRate = "auto",
                                       weightDecay = 1e-4,
                                       batchSize = 256,
                                       epochs = 2,
                                       seed = NULL,
                                       device = "cpu"),
      hyperParamSearch = "random",
      randomSample = 1
    )
) {
  embeddingFilePath <- normalizePath(embeddingFilePath)
  checkIsClass(embeddingFilePath, "character")
  checkFileExists(embeddingFilePath)
  
  
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  modelSettings$estimatorSettings$initStrategy <-
    reticulate::import_from_path("InitStrategy",
                                 path = path)$CustomEmbeddingInitStrategy()
  modelSettings$estimatorSettings$embeddingFilePath <- embeddingFilePath
  transformerSettings <- modelSettings

  attr(transformerSettings, "settings")$name <- "CustomEmbeddingModel"
  return(transformerSettings)
}
