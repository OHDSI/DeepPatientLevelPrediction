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
#' Create Model Settings with Custom Embeddings
#'
#' Configures a model to use supplied embeddings, such as Poincare embeddings
#' or embeddings from a foundation model.
#'
#' @param embeddingFilePath Path to a PyTorch file containing a dictionary with
#'   `concept_ids`, a PyTorch long tensor, and `embeddings`, a PyTorch float
#'   tensor.
#' @param modelSettings Settings for a model with an embedding layer named
#'   `embedding`. The supplied embeddings replace that layer.
#' @param embeddingsClass Embedding implementation, either
#'   `"CustomEmbeddings"` or `"PoincareEmbeddings"`.
#'
#' @return A `modelSettings` object configured to initialize custom embeddings.
#'
#' @examples
#' \dontrun{
#' # Requires a PyTorch embeddings file created outside this example.
#' modelSettings <- setCustomEmbeddingModel(
#'   embeddingFilePath = "path/to/embeddings.pt",
#'   modelSettings = setDefaultTransformer()
#' )
#' }
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
      dimHidden = 32,
      estimatorSettings = setEstimator(learningRate = "auto",
                                       weightDecay = 1e-4,
                                       batchSize = 256,
                                       epochs = 2,
                                       seed = NULL,
                                       device = "cpu"),
      hyperParamSearch = "random",
      randomSample = 1
    ),
    embeddingsClass = "CustomEmbeddings"
) {
  checkIsClass(embeddingFilePath, "character")
  checkFileExists(embeddingFilePath)
  embeddingFilePath <- normalizePath(embeddingFilePath, mustWork = TRUE)
  checkIsClass(embeddingsClass, "character")
  checkInStringVector(embeddingsClass, c("CustomEmbeddings", "PoincareEmbeddings"))
  
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  modelSettings$estimatorSettings$initStrategy <-
    reticulate::import_from_path("InitStrategy",
                                 path = path)$CustomEmbeddingInitStrategy(
                                   embedding_class = embeddingsClass,
                                   embedding_file = embeddingFilePath
                                 )
  transformerSettings <- modelSettings

  attr(transformerSettings, "settings")$name <- "CustomEmbeddingModel"
  return(transformerSettings)
}
