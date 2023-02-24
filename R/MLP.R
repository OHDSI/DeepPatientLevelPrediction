# @file MLP.R
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

#' setMultiLayerPerceptron
#'
#' @description
#' Creates settings for a Multilayer perceptron model
#'
#' @details
#' Model architecture
#'
#'
#' @param numLayers         Number of layers in network, default: 1:16
#' @param sizeHidden        Amount of neurons in each default layer, default: 2^(6:10) (64 to 1024)
#' @param dropout           How much dropout to apply after first linear, default: seq(0, 0.3, 0.05)
#' @param sizeEmbedding     Size of embedding layer, default: 2^(6:9) (64 to 512)
#' @param estimatorSettings settings of Estimator created with `setEstimator`
#' @param hyperParamSearch  Which kind of hyperparameter search to use random sampling or exhaustive grid search. default: 'random'
#' @param randomSample      How many random samples from hyperparameter space to use
#' @param randomSampleSeed  Random seed to sample hyperparameter combinations
#'
#' @export
setMultiLayerPerceptron <- function(numLayers = c(1:8),
                                    sizeHidden = c(2^(6:9)),
                                    dropout = c(seq(0, 0.5, 0.05)),
                                    sizeEmbedding = c(2^(6:9)),
                                    estimatorSettings = setEstimator(
                                      learningRate = 'auto',
                                      weightDecay = c(1e-6, 1e-3),
                                      batchSize = 1024,
                                      epochs = 30,
                                      device="cpu"),                              
                                    hyperParamSearch = "random",
                                    randomSample = 100,
                                    randomSampleSeed = NULL) {

  paramGrid <- list(
    numLayers = numLayers,
    sizeHidden = sizeHidden,
    dropout = dropout,
    sizeEmbedding = sizeEmbedding
  )
  
  paramGrid <- c(paramGrid, estimatorSettings$paramsToTune)
  
  param <- PatientLevelPrediction::listCartesian(paramGrid)
  if (randomSample>length(param)) {
    stop(paste("\n Chosen amount of randomSamples is higher than the amount of possible hyperparameter combinations.", 
               "\n randomSample:", randomSample,"\n Possible hyperparameter combinations:", length(param),
               "\n Please lower the amount of randomSamples"))
  }
  
  if (hyperParamSearch == "random") {
    suppressWarnings(withr::with_seed(randomSampleSeed, {param <- param[sample(length(param), randomSample)]}))
  }

  results <- list(
    fitFunction = "fitEstimator",
    param = param,
    estimatorSettings = estimatorSettings,
    modelType = "MLP",
    saveType = "file",
    modelParamNames = c(
      "numLayers", "sizeHidden",
      "dropout", "sizeEmbedding"
    )
  )

  class(results) <- "modelSettings"

  return(results)
}


MLP <- torch::nn_module(
  name = "MLP",
  initialize = function(catFeatures, numFeatures = 0, sizeEmbedding, sizeHidden, numLayers,
                        activation = torch::nn_relu,
                        normalization = torch::nn_batch_norm1d, dropout = NULL,
                        d_out = 1) {
    self$embedding <- torch::nn_embedding_bag(
      num_embeddings = catFeatures + 1,
      embedding_dim = sizeEmbedding,
      padding_idx = 1
    )
    if (numFeatures != 0) {
      self$numEmbedding <- numericalEmbedding(numFeatures, sizeEmbedding)
    }

    self$first_layer <- torch::nn_linear(sizeEmbedding, sizeHidden)


    self$layers <- torch::nn_module_list(lapply(
      1:numLayers,
      function(x) {
        MLPLayer(
          sizeHidden,
          normalization, activation,
          dropout
        )
      }
    ))
    self$lastNorm <- normalization(sizeHidden)
    self$head <- torch::nn_linear(sizeHidden, d_out)

    self$lastAct <- activation()
  },
  forward = function(x) {
    x_cat <- x$cat
    x_num <- x$num
    x_cat <- self$embedding(x_cat + 1L) # padding_idx is 1
    if (!is.null(x_num)) {
      x <- (x_cat + self$numEmbedding(x_num)$mean(dim = 2)) / 2
    } else {
      x <- x_cat
    }
    x <- self$first_layer(x)

    for (i in 1:length(self$layers)) {
      x <- self$layers[[i]](x)
    }
    x <- self$lastNorm(x)
    x <- self$lastAct(x)
    x <- self$head(x)
    x <- x$squeeze(-1)
    return(x)
  }
)

MLPLayer <- torch::nn_module(
  name = "MLPLayer",
  initialize = function(sizeHidden = 64,
                        normalization = torch::nn_batch_norm1d,
                        activation = torch::nn_relu,
                        dropout = 0.0, bias = TRUE) {
    self$norm <- normalization(sizeHidden)
    self$activation <- activation()
    self$linear <- torch::nn_linear(sizeHidden, sizeHidden, bias = bias)

    if (!is.null(dropout) | !dropout == 0.0) {
      self$dropout <- torch::nn_dropout(p = dropout)
    }
  },
  forward = function(x) {
    x <- self$linear(self$norm(x))
    if (!is.null(self$dropout)) {
      x <- self$dropout(x)
    }
    return(self$activation(x))
  }
)
