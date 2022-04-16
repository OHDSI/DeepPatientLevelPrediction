#' setTransformer
#'
#' @description A transformer model
#' @details from https://arxiv.org/abs/2106.11959 
#' @export
setTransformer <- function(numBlocks=3, dimToken=96, dimOut=1,
                           numHeads=8, attDropout=0.25, ffnDropout=0.25,
                           resDropout=0,dimHidden=512, weightDecay=1e-6, 
                           learningRate=3e-4, batchSize=1024,
                           epochs=10, device='cpu', hyperParamSearch='random',
                           randomSamples=100, seed=NULL) {
  if (!is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
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

  param <- listCartesian(paramGrid)

  if (hyperParamSearch=='random'){
    param <- param[sample(length(param), randomSamples)]
  }

  attr(param, 'settings') <- list(
    seed = seed[1],
    device = device,
    batchSize = batchSize,
    epochs = epochs,
    name = "Transformer",
    saveType = 'file',
    modelParamNames = c('numBlocks', 'dimToken', 'dimOut', 'numHeads',
                        'attDropout', 'ffnDropout', 'resDropout', 'dimHidden'),
    baseModel = 'Transformer'
  )

  results <- list(
    fitFunction = 'fitEstimator',
    param = param
  )

  class(results) <- 'modelSettings'
  return(results)
}


Transformer <- torch::nn_module(
  name='Transformer',
  initialize = function(catFeatures, numFeatures, numBlocks, dimToken, dimOut, 
                       numHeads, attDropout, ffnDropout, resDropout, 
                       headActivation=torch::nn_relu,
                       activation=torch::nn_relu,
                       ffnNorm=torch::nn_layer_norm, 
                       headNorm=torch::nn_layer_norm,
                       attNorm=torch::nn_layer_norm,
                       dimHidden){
    self$embedding <- Embedding(catFeatures, dimToken)
    dimToken <- dimToken + numFeatures # because I concatenate numerical features to embedding
    
    self$layers <- torch::nn_module_list(lapply(1:numBlocks,
                      function(x) {
                         layer <- torch::nn_module_list()
                         layer$add_module('attention', torch::nn_multihead_attention(dimToken,numHeads,
                                                                                     dropout=attDropout,
                                                                                     bias=TRUE))
                         layer$add_module('ffn', FeedForwardBlock(dimToken, dimHidden,
                                                                  biasFirst=TRUE,
                                                                  biasSecond=TRUE,
                                                                  dropout=ffnDropout,
                                                                  activation=activation))
                          layer$add_module('attentionResDropout', torch::nn_dropout(resDropout))      
                          layer$add_module('ffnResDropout', torch::nn_dropout(resDropout))
                          layer$add_module('ffnNorm', ffnNorm(dimToken))
              
                          if (x==1) {
                            layer$add_module('attentionNorm', attNorm(dimToken))
                        }
                         return(layer) 
                          }))
    self$head <- Head(dimToken, bias=TRUE, activation=headActivation, 
                      headNorm, dimOut)
  },
  forward = function(x_num, x_cat){
    x_cat <- self$embedding(x_cat)
    x <- torch::torch_cat(list(x_cat, x_num), dim=2L)
    for (i in 1:length(self$layers)) {
      layer <- self$layers[[i]]
      xResidual <- self$startResidual(layer, 'attention', x)

      if (i==length(self$layers)) {
        xResidual <- layer$attention(xResidual[,-1], xResidual) # in final layer take only attention on CLS token
        x <- x[,-1]
        } else {
        xResidual <- layer$attention(xResidual, xResidual)
        }
      x <- self$endResidual(layer, 'attention', x, xResidual)

      xResidual <- self$startResidual(layer, 'ffn', x, xResidual)
      xResidual <- layer$ffn(xResidual)
      x <- self$endResidual(layer, 'ffn', x, xResidual)
    }
    x <- self$head(x)
    return(x)
  },
  startResidual = function(layer, stage, x) {
    xResidual <- x
    normKey <- paste0(stage, 'Norm')
    if (normKey %in% names(as.list(layer))) {
      xResidual <- layer[[normKey]](xResidual)
    }
    return(xResidual)
  },
  endResidual = function(layer, stage, x, xResidual) {
    dropoutKey <- paste0(stage, 'ResDropout')
    xResidual <-layer$dropoutKey(xResidual)
    x <- x + xResidual
    return(x)
  }
)


FeedForwardBlock <- torch::nn_module(
  name='FeedForwardBlock',
  initialize = function(dimToken, dimHidden, biasFirst, biasSecond,
                        dropout, activation) {
    self$linearFirst <- torch::nn_linear(dimToken, dimHidden, biasFirst)
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
  name='Head',
  initialize = function(dimIn, bias, activation, normalization, dimOut) {
    self$normalization <- normalization(dimIn)
    self$activation <- activation()
    self$linear <- torch::nn_linear(dimIn,dimOut, bias)
  },
  forward = function(x) {
    x <- x[,-1] # ?
    x <- self$normalization(x)
    x <- self$activation(x)
    x <- self$linear(x)
    return(x)
  }
)

Embedding <- torch::nn_module(
  name='Embedding',
  initialize = function(numEmbeddings, embeddingDim) {
    self$embedding <- torch::nn_embedding(numEmbeddings, embeddingDim)
    categoryOffsets <- torch::torch_arange(1, numEmbeddings)
    self$register_buffer('categoryOffsets', categoryOffsets, persistent=FALSE)
  },
  forward = function(x_cat) {
    x <- self$embedding(x_cat * self$categoryOffsets)
    }
)
