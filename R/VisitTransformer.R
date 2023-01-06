# @file VisitTransformer.R
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

#' create settings for a transformer trained using temporal
#'
#' @description A transformer model that aggregates embeddings per visit
#' @details from https://arxiv.org/abs/2007.05611
#'
#' @param embeddingDim            embedding dimension (size of token)
#' @param numHeads                number of attention heads
#' @param numHidden               size of hidden layers after attention
#' @param numLayers               How many layers in model
#' @param parallelPools           How many pools used to aggregate before prediction head (see paper)
#' @param ffnDropout              dropout to use in feedforward block
#' @param attentionDropout        dropout to use on attentions
#' @param residualDropout              dropout to use in residual connections
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
setVisitTransformer <- function(embeddingDim=64,
                                numHeads=4,
                                numHidden=512,
                                numLayers=4,
                                parallelPools=10,
                                ffnDropout=0.05,
                                attentionDropout=0.05,
                                residualDropout=0.05,
                                weightDecay=1e-6,
                                learningRate=3e-4,
                                batchSize=1024,
                                epochs=10,
                                device='cpu',
                                hyperParamSearch = 'random',
                                randomSample=1,
                                seed = NULL
                                ) { 
  
  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  
  if (embeddingDim %% numHeads != 0) {
    stop(paste(
      "embeddingDim needs to divisble by numHeads. embeddingDim =", embeddingDim,
      "is not divisible by numHeads =", numHeads
    ))
  }
  
  paramGrid <- list(
    numLayers = numLayers,
    embeddingDim = embeddingDim,
    numHeads = numHeads,
    numHidden = numHidden,
    attentionDropout = attentionDropout,
    ffnDropout = ffnDropout,
    residualDropout = residualDropout,
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
    name = "VisitTransformer",
    saveType = "file",
    modelParamNames = c(
      "numLayers", "embeddingDim", "numHeads",
      "attentionDropout", "ffnDropout", "residualDropout", "numHidden"
    ),
    baseModel = "VisitTransformer",
    datasetCreator = TemporalDataset
  )
  
  results <- list(
    fitFunction = "fitEstimator",
    param = param
  )
  
  class(results) <- "modelSettings"
  return(results)
}

VisitTransformer <- torch::nn_module(
  name = 'VisitTransformer',
  initialize = function(temporalFeatures, 
                        staticFeatures=4,
                        maxTime=365,  # max timeId of sequence
                        maxVisits=40,   # max number of discrete times in sequence
                        embeddingDim,
                        numHeads,
                        numHidden,
                        numLayers,
                        parallelPools=10,
                        ffnDropout=0.05,
                        attentionDropout=0.05,
                        residualDropout=0.05
                        ) {
    # self$embedding <- torch::nn_linear(temporalFeatures, embeddingDim - staticFeatures, bias=FALSE)
    self$embedding <- SparseLinear(temporalFeatures, embeddingDim - staticFeatures)
    self$posEncoding <- PositionalEncoding(embeddingDim - staticFeatures, maxTime)
    
    self$ffnDropout <- ffnDropout
    self$residualDropout <- residualDropout
    self$layers <- torch::nn_module_list(lapply(
      1:numLayers,
      function(x) {
        layer <- torch::nn_module_list()
        layer$add_module("attention", torch::nn_multihead_attention(embeddingDim, 
                                                                    numHeads,
                                                                    dropout = attentionDropout,
                                                                    bias = TRUE,
                                                                    batch_first = TRUE)
                        )
        layer$add_module("linear0", torch::nn_linear(embeddingDim,
                                                     numHidden * 2)
                        )
        layer$add_module("linear1", torch::nn_linear(numHidden,
                                                     embeddingDim)
                        )
        layer$add_module("norm1", torch::nn_layer_norm(embeddingDim)
                         )  
        
        if (x != 1) {
          layer$add_module("norm0", torch::nn_layer_norm(embeddingDim)
                           )
        }
        return(layer)
      }
    ))
    
    self$activation <- nn_reglu()
    self$lastActivation <- torch::nn_relu()
    self$embeddingDim <- embeddingDim
    self$parallelPools <- parallelPools
    self$staticFeatures <- staticFeatures
    
    self$pooler <- torch::nn_linear(maxVisits, parallelPools)
    self$predictionHead <- torch::nn_linear((embeddingDim) * parallelPools, 1)

    self$maxVisits <- maxVisits
    
    self$initWeights()
  },
  initWeights = function() {
    initRange <- 0.1
    torch::nn_init_uniform_(self$embedding$weight, -initRange, initRange)
  },
  forward = function(input) {
    # rescale embeddings with sqrt(embeddingDim) , possible better to downscale timeEmbedding?
    dataEmbedding <- self$embedding(input$sequences) * sqrt(self$embeddingDim)
    timeEmbedding <- self$posEncoding(input$visits)
    embedding <- dataEmbedding + timeEmbedding
    # concat staticData to embeddings
    x <- torch::torch_cat(list(embedding, 
                               input$static$to_dense()[,NULL,][["repeat"]](c(1,self$maxVisits,1))),
                          dim=3)
    # generate key padding mask
    mask <- torch::torch_arange(1, self$maxVisits, 
                                device = input$sequences$device)[NULL,] >
      input$lengths[,NULL]
    for (i in 1:length(self$layers)) {
      isLastLayer <- i == length(self$layers)
      layer <- self$layers[[i]]
      
      xResidual <- self$startResidual(x, layer, 0)
      xResidual <- layer[['attention']](xResidual,
                                        xResidual,
                                        xResidual,
                                        key_padding_mask=mask)[[1]]
      
      if (isLastLayer) {
        x <- x[, 1:xResidual$shape[[2]]]
      }
      x <- self$endResidual(x, xResidual, layer)
      
      xResidual <- self$startResidual(x, layer, 1)
      xResidual <- layer[['linear0']](xResidual)
      xResidual <- self$activation(xResidual)
      
      if (self$ffnDropout > 0.0) {
        xResidual <- torch::nnf_dropout(xResidual, self$ffnDropout, self$training)
      }
      xResidual <- layer[['linear1']](xResidual)
      x <- self$endResidual(x, xResidual, layer)
    }
    output <- self$pooler(x$transpose(2,3))$reshape(c(-1, self$parallelPools * self$embeddingDim))
    output <- self$predictionHead(output)
    return(output$squeeze())
  },
  startResidual = function(x, layer, normIndex) {
    xResidual <- x
    normKey <- paste0("norm", normIndex)
    if (normKey %in% names(as.list(layer))) {
      xResidual <- layer[[normKey]](xResidual)
    }
    return(xResidual)
  },
  endResidual = function(x, xResidual, layer) {
    if (self$residualDropout>0.0) {
      xResidual <- torch::nnf_dropout(xResidual, self$residualDropout, 
                                      self$training)
    }
    x <- x + xResidual
    return(x)
  }
)

PositionalEncoding <- torch::nn_module(
  name = 'PositionalEncoding',
  initialize = function(embeddingDim, maxLength=1000) {
    pe <- torch::torch_zeros(maxLength, embeddingDim)
    position <- torch::torch_arange(0, maxLength - 1, dtype = torch::torch_float32())$unsqueeze(2)
    divTerm <- torch::torch_exp(torch::torch_arange(0, embeddingDim - 1, 2) * (-log(10000) / embeddingDim))
    pe[, seq(1, pe$shape[[2]], 2)] <- torch::torch_sin(position * divTerm)
    pe[, seq(2, pe$shape[[2]], 2)] <- torch::torch_cos(position * divTerm)[,1:floor(embeddingDim/2)]
    pe <- pe$unsqueeze(1)$transpose(1, 2)
    
    self$register_buffer('pe', pe)
  },
  forward = function(x) {
    x <- torch::nnf_embedding(x, self$pe$squeeze(), padding_idx = 1)
  }
)

SparseLinear <- torch::nn_module(
  name = 'SparseLinear',
  initialize = function(inFeatures, outFeatures) {
      self$inFeatures <- inFeatures
      self$outFeatures <- outFeatures
      self$weight <- torch::nn_parameter(torch::torch_empty(inFeatures, outFeatures))
      self$resetParameters()
  },
  resetParameters = function() {
    torch::nn_init_kaiming_uniform_(self$weight, a=sqrt(5))
  },
  forward = function(x) {
    return(torch::torch_bmm(x, self$weight$unsqueeze(1)$expand(c(x$shape[[1]],-1,-1))))
  }
)

# modelled after Karpathy's nanoGPT attention module
CustomSelfAttention <- torch::nn_module(
  name = 'CustomSelfAttention',
  initialize = function() {
    
    
  },
  forward = function() {
    
  }
)




