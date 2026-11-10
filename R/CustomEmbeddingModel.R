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
#'   tensor. Binary covariates whose concept ID is in the file get its vector.
#'   Numeric covariates, such as measurement values, keep their regular
#'   embedding, which is scaled by the value, because a concept vector does not
#'   carry a value.
#' @param modelSettings Settings for a model with an embedding layer named
#'   `embedding`. The supplied embeddings replace that layer.
#' @param embeddingsClass Embedding implementation, either
#'   `"CustomEmbeddings"` or `"PoincareEmbeddings"`.
#' @param curvature For `"PoincareEmbeddings"`, the magnitude `c` of the
#'   curvature `-c` of the Poincare ball that the embeddings were trained in.
#'   The norm of every embedding must be below `1 / sqrt(c)`. This is the
#'   curvature itself, not the parameter of a library that transforms it:
#'   HypLL, used by OHDSI/context, applies softplus, so an untrained curvature
#'   parameter of 1 there means `c = log(1 + exp(1))`, about 1.313. If the
#'   curvature was trained, use its final value after softplus.
#'
#' @return A `modelSettings` object configured to initialize custom embeddings.
#'
#' @examples
#' \dontrun{
#' # Requires Python and the package's Python dependencies, including PyTorch.
#' # See vignette("Installing") for setup instructions.
#' embeddingFilePath <- tempfile(fileext = ".pt")
#'
#' # Toy embeddings: one row per concept ID, with eight values per row.
#' conceptIds <- c(1L, 2L, 3L)
#' embeddings <- list(
#'   concept_ids = torch$tensor(conceptIds, dtype = torch$long),
#'   embeddings = torch$randn(length(conceptIds), 8L, dtype = torch$float)
#' )
#' torch$save(embeddings, embeddingFilePath)
#'
#' modelSettings <- setCustomEmbeddingModel(
#'   embeddingFilePath = embeddingFilePath,
#'   modelSettings = setDefaultTransformer()
#' )
#'
#' # Clean up this example's file. In real use, keep it for as long as you use
#' # the model: modelSettings and the saved model store its path, not the
#' # embedding tensors, and predicting with a saved model reads it again.
#' unlink(embeddingFilePath)
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
    embeddingsClass = "CustomEmbeddings",
    curvature = 1
) {
  checkIsClass(embeddingFilePath, "character")
  checkFileExists(embeddingFilePath)
  embeddingFilePath <- normalizePath(embeddingFilePath, mustWork = TRUE)
  checkIsClass(embeddingsClass, "character")
  checkInStringVector(embeddingsClass, c("CustomEmbeddings", "PoincareEmbeddings"))
  checkIsClass(curvature, c("numeric", "integer"))
  if (length(curvature) != 1 || !is.finite(curvature) || curvature <= 0) {
    stop("curvature must be a single positive number")
  }
  if (embeddingsClass != "PoincareEmbeddings" && curvature != 1) {
    stop("curvature only applies to embeddingsClass = \"PoincareEmbeddings\"")
  }
  
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  modelSettings$estimatorSettings$initStrategy <-
    reticulate::import_from_path("InitStrategy",
                                 path = path)$CustomEmbeddingInitStrategy(
                                   embedding_class = embeddingsClass,
                                   embedding_file = embeddingFilePath,
                                   curvature = curvature
                                 )
  transformerSettings <- modelSettings

  attr(transformerSettings, "settings")$name <- "CustomEmbeddingModel"
  return(transformerSettings)
}
