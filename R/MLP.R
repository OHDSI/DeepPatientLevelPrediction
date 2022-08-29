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
#' @param weightDecay       Weight decay to apply, default: c(1e-6, 1e-3)
#' @param learningRate      Learning rate to use. default: c(1e-2, 1e-5)
#' @param seed              Seed to use for sampling hyperparameter space
#' @param hyperParamSearch  Which kind of hyperparameter search to use random sampling or exhaustive grid search. default: 'random'
#' @param randomSample      How many random samples from hyperparameter space to use
#' @param device            Which device to run analysis on, either 'cpu' or 'cuda', default: 'cpu'
#' @param batchSize         Size of batch, default: 1024
#' @param epochs            Number of epochs to run, default: 10
#'
#' @export
setMultiLayerPerceptron <- function(numLayers = c(1:8),
                                    sizeHidden = c(2^(6:9)),
                                    dropout = c(seq(0, 0.5, 0.05)),
                                    sizeEmbedding = c(2^(6:9)),
                                    weightDecay = c(1e-6, 1e-3),
                                    learningRate = c(1e-2, 3e-4, 1e-5),
                                    seed = NULL,
                                    hyperParamSearch = "random",
                                    randomSample = 100,
                                    device = "cpu",
                                    batchSize = 1024,
                                    epochs = 30) {
  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }

  paramGrid <- list(
    numLayers = numLayers,
    sizeHidden = sizeHidden,
    dropout = dropout,
    sizeEmbedding = sizeEmbedding,
    weightDecay = weightDecay,
    learningRate = learningRate,
    seed = list(as.integer(seed[[1]]))
  )

  param <- PatientLevelPrediction::listCartesian(paramGrid)

  if (hyperParamSearch == "random") {
    param <- param[sample(length(param), randomSample)]
  }

  attr(param, "settings") <- list(
    seed = seed[1],
    device = device,
    batchSize = batchSize,
    epochs = epochs,
    name = "MLP",
    saveType = "file",
    modelParamNames = c(
      "numLayers", "sizeHidden",
      "dropout", "sizeEmbedding"
    ),
    baseModel = "MLP"
  )

  results <- list(
    fitFunction = "fitEstimator",
    param = param
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
