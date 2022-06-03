# @file ResNet.R
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

#' setResNet
#'
#' @description 
#' Creates settings for a ResNet model
#' 
#' @details
#' Model architecture from by https://arxiv.org/abs/2106.11959 
#' 
#' 
#' @param numLayers         Number of layers in network, default: 1:16
#' @param sizeHidden        Amount of neurons in each default layer, default: 2^(6:10) (64 to 1024)
#' @param hiddenFactor      How much to grow the amount of neurons in each ResLayer, default: 1:4
#' @param residualDropout   How much dropout to apply after last linear layer in ResLayer, default: seq(0, 0.3, 0.05)
#' @param hiddenDropout     How much dropout to apply after first linear layer in ResLayer, default: seq(0, 0.3, 0.05)
#' @param normalization     Which type of normalization to use. Default: 'Batchnorm'
#' @param activation        What kind of activation to use. Default: 'RelU'
#' @param sizeEmbedding     Size of embedding layer, default: 2^(6:9) (64 to 512)
#' @param weightDecay       Weight decay to apply, default: c(1e-6, 1e-3)
#' @param learningRate      Learning rate to use. default: c(1e-2, 1e-5)
#' @param seed              Seed to use for sampling hyperparameter space
#' @param hyperParamSearch  Which kind of hyperparameter search to use random sampling or exhaustive grid search. default: 'random'
#' @param randomSample      How many random samples from hyperparameter space to use
#' @param device            Which device to run analysis on, either 'cpu' or 'cuda', default: 'cpu'
#' @param batchSize        Size of batch, default: 1024
#' @param epochs            Number of epochs to run, default: 10
#'
#' @export
setResNet <- function(
  numLayers = c(1:8), 
  sizeHidden = c(2^(6:9)), 
  hiddenFactor = c(1:4),
  residualDropout = c(seq(0,0.5,0.05)), 
  hiddenDropout = c(seq(0,0.5,0.05)),
  normalization = c('BatchNorm'), 
  activation = c('RelU'),
  sizeEmbedding = c(2^(6:9)), 
  weightDecay = c(1e-6, 1e-3),
  learningRate = c(1e-2, 3e-4, 1e-5), 
  seed = NULL, 
  hyperParamSearch = 'random',
  randomSample = 100, 
  device = 'cpu', 
  batchSize = 1024, 
  epochs = 30
  ) {

  if (is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  
  paramGrid <- list(
    numLayers = numLayers, 
    sizeHidden = sizeHidden,
    hiddenFactor = hiddenFactor,
    residualDropout = residualDropout,
    hiddenDropout = hiddenDropout,
    sizeEmbedding = sizeEmbedding, 
    weightDecay = weightDecay,
    learningRate = learningRate,
    seed = list(as.integer(seed[[1]]))
  )
  
  param <- listCartesian(paramGrid)
  
  if (hyperParamSearch=='random'){
    param <- param[sample(length(param), randomSample)]
  }

  attr(param, 'settings') <- list(
    seed = seed[1],
    device = device,
    batchSize = batchSize,
    epochs = epochs,
    name = "ResNet",
    saveType = 'file',
    modelParamNames = c("numLayers", "sizeHidden", "hiddenFactor",
                         "residualDropout", "hiddenDropout", "sizeEmbedding"),
    baseModel = 'ResNet'
  )

  results <- list(
    fitFunction = 'fitEstimator',
    param = param
  )

  class(results) <- 'modelSettings'

  return(results)

}

ResNet <- torch::nn_module(
  name='ResNet',
  initialize=function(catFeatures, numFeatures=0, sizeEmbedding, sizeHidden, numLayers,
                      hiddenFactor, activation=torch::nn_relu, 
                      normalization=torch::nn_batch_norm1d, hiddenDropout=NULL,
                      residualDropout=NULL, d_out=1) {
    # self$embedding <- EmbeddingBag(numEmbeddings=catFeatures + 1L, 
    #                                embeddingDim=sizeEmbedding,
    #                                paddingIdx=1)
    self$embedding <- torch::nn_embedding_bag(num_embeddings = catFeatures + 1,
                                              embedding_dim = sizeEmbedding,
                                              padding_idx = 1)
    self$first_layer <- torch::nn_linear(sizeEmbedding + numFeatures, sizeHidden)
    
    resHidden <- sizeHidden * hiddenFactor
    
    self$layers <- torch::nn_module_list(lapply(1:numLayers,
                                                 function (x) ResLayer(sizeHidden, resHidden,
                                                          normalization, activation,
                                                          hiddenDropout,
                                                          residualDropout)))
    self$lastNorm <- normalization(sizeHidden)
    self$head <- torch::nn_linear(sizeHidden, d_out)
    
    self$lastAct <- activation()
    
  },
      
  forward=function(x) {
    x_cat <- x$cat
    x_num <- x$num
    x_cat <- self$embedding(x_cat + 1L) # padding_idx is 1
    if (!is.null(x_num)) {
      x <- torch::torch_cat(list(x_cat, x_num), dim=2L)
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

ResLayer <- torch::nn_module(
  name='ResLayer',
  
  initialize=function(sizeHidden, resHidden, normalization,
                      activation, hiddenDropout=NULL, residualDropout=NULL){
    self$norm <- normalization(sizeHidden)
    self$linear0 <- torch::nn_linear(sizeHidden, resHidden)
    self$linear1 <- torch::nn_linear(resHidden, sizeHidden)
    
    self$activation <- activation
    if (!is.null(hiddenDropout)){
      self$hiddenDropout <- torch::nn_dropout(p=hiddenDropout)
    }
    if (!is.null(residualDropout)) 
    {
      self$residualDropout <- torch::nn_dropout(p=residualDropout)
    }
    
    self$activation <- activation()
    
  },
  
  forward=function(x) {
    z <- x
    z <- self$norm(z)
    z <- self$linear0(z)
    z <- self$activation(z)
    if (!is.null(self$hiddenDropout)) {
      z <- self$hiddenDropout(z)
    }
    z <- self$linear1(z)
    if (!is.null(self$residualDropout)) {
      z <- self$residualDropout(z)
    }
    x <- z + x 
    return(x)
  }
)


listCartesian <- function(allList){
  
  sizes <- lapply(allList, function(x) 1:length(x))
  combinations <- expand.grid(sizes)
  
  result <- list()
  length(result) <- nrow(combinations)
  
  for(i in 1:nrow(combinations)){
    tempList <- list()
    for(j in 1:ncol(combinations)){
      tempList <- c(tempList, list(allList[[j]][combinations[[i,j]]]))
    }
    names(tempList) <- names(allList)
    result[[i]] <- tempList
  }
  
  return(result)
}


# export this in PLP
computeGridPerformance <- PatientLevelPrediction:::computeGridPerformance

