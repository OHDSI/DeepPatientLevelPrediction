# @file Transformer.R
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

#' Create default settings for a non-temporal transformer
#' 
#' @description A transformer model with default hyperparameters
#' @details from https://arxiv.org/abs/2106.11959
#' Default hyperparameters from paper
#' @param device   Which device to run analysis on, either 'cpu' or 'cuda', default: 'cpu'  
#' @param batchSize Size of batch, default: 512
#' @param epochs  Number of epochs to run, default: 10
#' @param seed    random seed to use
#' 
#' @export
setDefaultTransformer <- function(device='cpu',
                                  batchSize=512,
                                  epochs=10,
                                  seed=NULL) {
  transformerSettings <- setTransformer(numBlocks = 3,
                                        dimToken = 192,
                                        dimOut = 1,
                                        numHeads = 8,
                                        attDropout = 0.2,
                                        ffnDropout = 0.1,
                                        resDropout = 0.0,
                                        dimHidden = 256,
                                        weightDecay = 1e-5,
                                        learningRate = 1e-4,
                                        batchSize = batchSize,
                                        epochs = epochs,
                                        device = device,
                                        hyperParamSearch = 'random',
                                        randomSample = 1,
                                        seed = seed)
  attr(transformerSettings, 'settings')$name <- 'defaultTransformer'
  return(transformerSettings)
}

#' create settings for training a non-temporal transformer
#'
#' @description A transformer model
#' @details from https://arxiv.org/abs/2106.11959
#'
#' @param numBlocks               number of transformer blocks
#' @param dimToken                dimension of each token (embedding size)
#' @param dimOut                  dimension of output, usually 1 for binary problems
#' @param numHeads                number of attention heads
#' @param attDropout              dropout to use on attentions
#' @param ffnDropout              dropout to use in feedforward block
#' @param resDropout              dropout to use in residual connections
#' @param dimHidden               dimension of the feedworward block
#' @param weightDecay             weightdecay to use
#' @param learningRate            learning rate to use
#' @param batchSize               batchSize to use
#' @param epochs                  How many epochs to run the model for
#' @param device                  Which device to use, cpu or cuda
#' @param hyperParamSearch        what kind of hyperparameter search to do, default 'random'
#' @param randomSample            How many samples to use in hyperparameter search if random
#' @param seed                    Random seed to use
#'
#' @export
setTransformer <- function(numBlocks = 3, dimToken = 96, dimOut = 1,
                           numHeads = 8, attDropout = 0.25, ffnDropout = 0.25,
                           resDropout = 0, dimHidden = 512, weightDecay = 1e-6,
                           learningRate = 3e-4, batchSize = 1024,
                           epochs = 10, device = "cpu", hyperParamSearch = "random",
                           randomSample = 1, seed = NULL) {
  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }

  if (dimToken %% numHeads != 0) {
    stop(paste(
      "dimToken needs to divisble by numHeads. dimToken =", dimToken,
      "is not divisible by numHeads =", numHeads
    ))
  }

  paramGrid <- list(
    numBlocks = numBlocks,
    dimToken = dimToken,
    dimOut = dimOut,
    numHeads = numHeads,
    dimHidden = dimHidden,
    attDropout = attDropout,
    ffnDropout = ffnDropout,
    resDropout = resDropout,
    weightDecay = weightDecay,
    learningRate = learningRate,
    seed = list(as.integer(seed[[1]]))
  )

  param <- PatientLevelPrediction::listCartesian(paramGrid)

  if (randomSample>length(param)) {
    stop(paste("\n Chosen amount of randomSamples is higher than the amount of possible hyperparameter combinations.", 
               "\n randomSample:", randomSample,"\n Possible hyperparameter combinations:", length(param),
               "\n Please lower the amount of randomSample"))
  }
  
  if (hyperParamSearch == "random") {
    param <- param[sample(length(param), randomSample)]
  }

  attr(param, "settings") <- list(
    seed = seed[1],
    device = device,
    batchSize = batchSize,
    epochs = epochs,
    name = "Transformer",
    saveType = "file",
    modelParamNames = c(
      "numBlocks", "dimToken", "dimOut", "numHeads",
      "attDropout", "ffnDropout", "resDropout", "dimHidden"
    ),
    baseModel = "Transformer",
    datasetCreator = Dataset
  )

  results <- list(
    fitFunction = "fitEstimator",
    param = param
  )

  class(results) <- "modelSettings"
  return(results)
}


Transformer <- torch::nn_module(
  name = "Transformer",
  initialize = function(catFeatures, numFeatures, numBlocks, dimToken, dimOut = 1,
                        numHeads, attDropout, ffnDropout, resDropout,
                        headActivation = torch::nn_relu,
                        activation = NULL,
                        ffnNorm = torch::nn_layer_norm,
                        headNorm = torch::nn_layer_norm,
                        attNorm = torch::nn_layer_norm,
                        dimHidden) {
    activation <- nn_reglu
    self$Categoricalembedding <- Embedding(catFeatures + 1, dimToken) # + 1 for padding idx
    self$numericalEmbedding <- numericalEmbedding(numFeatures, dimToken)
    self$classToken <- ClassToken(dimToken)

    self$layers <- torch::nn_module_list(lapply(
      1:numBlocks,
      function(x) {
        layer <- torch::nn_module_list()
        layer$add_module("attention", torch::nn_multihead_attention(dimToken, numHeads,
          dropout = attDropout,
          bias = TRUE
        ))
        layer$add_module("ffn", FeedForwardBlock(dimToken, dimHidden,
          biasFirst = TRUE,
          biasSecond = TRUE,
          dropout = ffnDropout,
          activation = activation
        ))
        layer$add_module("attentionResDropout", torch::nn_dropout(resDropout))
        layer$add_module("ffnResDropout", torch::nn_dropout(resDropout))
        layer$add_module("ffnNorm", ffnNorm(dimToken))

        if (x != 1) {
          layer$add_module("attentionNorm", attNorm(dimToken))
        }
        return(layer)
      }
    ))
    self$head <- Head(dimToken,
      bias = TRUE, activation = headActivation,
      headNorm, dimOut
    )
  },
  forward = function(x) {
    mask <- torch::torch_where(x$cat == 0, TRUE, FALSE)
    input <- x
    cat <- self$Categoricalembedding(x$cat)
    if (!is.null(input$num)) {
      num <- self$numericalEmbedding(input$num)
      x <- torch::torch_cat(list(cat, num), dim = 2L)
      mask <- torch::torch_cat(list(mask, torch::torch_zeros(c(
        x$shape[1],
        num$shape[2]
      ),
      device = mask$device,
      dtype = mask$dtype
      )),
      dim = 2L
      )
    } else {
      x <- cat
    }
    x <- self$classToken(x)
    mask <- torch::torch_cat(list(mask, torch::torch_zeros(c(x$shape[1], 1),
      device = mask$device,
      dtype = mask$dtype
    )),
    dim = 2L
    )
    for (i in 1:length(self$layers)) {
      layer <- self$layers[[i]]
      xResidual <- self$startResidual(layer, "attention", x)

      if (i == length(self$layers)) {
        dims <- xResidual$shape
        # in final layer take only attention on CLS token
        xResidual <- layer$attention(
          xResidual[, -1]$view(c(dims[1], 1, dims[3]))$transpose(1, 2),
          xResidual$transpose(1, 2),
          xResidual$transpose(1, 2), mask
        )
        attnWeights <- xResidual[[2]]
        xResidual <- xResidual[[1]]
        x <- x[, -1]$view(c(dims[1], 1, dims[3]))
      } else {
        # attention input is seq_length x batch_size x embedding_dim
        xResidual <- layer$attention(
          xResidual$transpose(1, 2),
          xResidual$transpose(1, 2),
          xResidual$transpose(1, 2),
          mask,
        )[[1]]
      }
      x <- self$endResidual(layer, "attention", x, xResidual$transpose(1, 2))

      xResidual <- self$startResidual(layer, "ffn", x)
      xResidual <- layer$ffn(xResidual)
      x <- self$endResidual(layer, "ffn", x, xResidual)
    }
    x <- self$head(x)[, 1] # remove singleton dimension
    return(x)
  },
  startResidual = function(layer, stage, x) {
    xResidual <- x
    normKey <- paste0(stage, "Norm")
    if (normKey %in% names(as.list(layer))) {
      xResidual <- layer[[normKey]](xResidual)
    }
    return(xResidual)
  },
  endResidual = function(layer, stage, x, xResidual) {
    dropoutKey <- paste0(stage, "ResDropout")
    xResidual <- layer[[dropoutKey]](xResidual)
    x <- x + xResidual
    return(x)
  }
)


FeedForwardBlock <- torch::nn_module(
  name = "FeedForwardBlock",
  initialize = function(dimToken, dimHidden, biasFirst, biasSecond,
                        dropout, activation) {
    self$linearFirst <- torch::nn_linear(dimToken, dimHidden * 2, biasFirst)
    self$activation <- activation()
    self$dropout <- torch::nn_dropout(dropout)
    self$linearSecond <- torch::nn_linear(dimHidden, dimToken, biasSecond)
  },
  forward = function(x) {
    x <- self$linearFirst(x)
    x <- self$activation(x)
    x <- self$dropout(x)
    x <- self$linearSecond(x)
    return(x)
  }
)

Head <- torch::nn_module(
  name = "Head",
  initialize = function(dimIn, bias, activation, normalization, dimOut) {
    self$normalization <- normalization(dimIn)
    self$activation <- activation()
    self$linear <- torch::nn_linear(dimIn, dimOut, bias)
  },
  forward = function(x) {
    x <- x[, -1] # ?
    x <- self$normalization(x)
    x <- self$activation(x)
    x <- self$linear(x)
    return(x)
  }
)

Embedding <- torch::nn_module(
  name = "Embedding",
  initialize = function(numEmbeddings, embeddingDim) {
    self$embedding <- torch::nn_embedding(numEmbeddings, embeddingDim, padding_idx = 1)
  },
  forward = function(x_cat) {
    x <- self$embedding(x_cat + 1L) # padding idx is 1L
    return(x)
  }
)

numericalEmbedding <- torch::nn_module(
  name = "numericalEmbedding",
  initialize = function(numEmbeddings, embeddingDim, bias = TRUE) {
    self$weight <- torch::nn_parameter(torch::torch_empty(numEmbeddings, embeddingDim))
    if (bias) {
      self$bias <- torch::nn_parameter(torch::torch_empty(numEmbeddings, embeddingDim))
    } else {
      self$bias <- NULL
    }

    for (parameter in list(self$weight, self$bias)) {
      if (!is.null(parameter)) {
        torch::nn_init_kaiming_uniform_(parameter, a = sqrt(5))
      }
    }
  },
  forward = function(x) {
    x <- self$weight$unsqueeze(1) * x$unsqueeze(-1)
    if (!is.null(self$bias)) {
      x <- x + self$bias$unsqueeze(1)
    }
    return(x)
  }
)

# adds a class token embedding to embeddings
ClassToken <- torch::nn_module(
  name = "ClassToken",
  initialize = function(dimToken) {
    self$weight <- torch::nn_parameter(torch::torch_empty(dimToken, 1))
    torch::nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
  },
  expand = function(dims) {
    newDims <- vector("integer", length(dims) - 1) + 1
    return(self$weight$view(c(newDims, -1))$expand(c(dims, -1)))
  },
  forward = function(x) {
    return(torch::torch_cat(c(x, self$expand(c(dim(x)[[1]], 1))), dim = 2))
  }
)

nn_reglu <- torch::nn_module(
  name = "ReGlu",
  forward = function(x) {
    return(reglu(x))
  }
)


reglu <- function(x) {
  chunks <- x$chunk(2, dim = -1)

  return(chunks[[1]] * torch::nnf_relu(chunks[[2]]))
}
