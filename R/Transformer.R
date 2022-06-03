#' setTransformer
#'
#' @description A transformer model
#' @details from https://arxiv.org/abs/2106.11959 
#' 
#' @param numBlocks        TODO: add desc
#' @param dimToken        TODO: add desc
#' @param dimOut        TODO: add desc
#' @param numHeads        TODO: add desc
#' @param attDropout        TODO: add desc
#' @param ffnDropout        TODO: add desc
#' @param resDropout        TODO: add desc
#' @param dimHidden        TODO: add desc
#' @param weightDecay        TODO: add desc
#' @param learningRate        TODO: add desc
#' @param batchSize       TODO: add desc
#' @param epochs        TODO: add desc
#' @param device        TODO: add desc
#' @param hyperParamSearch       TODO: add desc
#' @param randomSamples       TODO: add desc
#' @param seed        TODO: add desc
#' 
#' @export
setTransformer <- function(
  numBlocks=3, dimToken=96, dimOut=1,
  numHeads=8, attDropout=0.25, ffnDropout=0.25,
  resDropout=0,dimHidden=512, weightDecay=1e-6, 
  learningRate=3e-4, batchSize=1024,
  epochs=10, device='cpu', hyperParamSearch='random',
  randomSamples=100, seed=NULL
) {
  if (!is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }
  
  if (dimToken %% numHeads != 0) {
    stop(paste('dimToken needs to divisble by numHeads. dimToken =', dimToken,
               'is not divisible by numHeads =', numHeads))
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
  initialize = function(catFeatures, numFeatures, numBlocks, dimToken, dimOut=1, 
                       numHeads, attDropout, ffnDropout, resDropout, 
                       headActivation=torch::nn_relu,
                       activation=torch::nn_relu,
                       ffnNorm=torch::nn_layer_norm, 
                       headNorm=torch::nn_layer_norm,
                       attNorm=torch::nn_layer_norm,
                       dimHidden){
    self$embedding <- Embedding(catFeatures + 1, dimToken) # + 1 for padding idx
    # self$numericalEmbedding <- numericalEmbedding(numFeatures, dimToken)
    self$classToken <- ClassToken(dimToken)
    
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
              
                          if (x!=1) {
                            layer$add_module('attentionNorm', attNorm(dimToken))
                        }
                         return(layer) 
                          }))
    self$head <- Head(dimToken, bias=TRUE, activation=headActivation, 
                      headNorm, dimOut)
  },
  forward = function(x){
    x_cat <- x$cat
    x_num <- x$num
    x <- self$embedding(x_cat)
    if (!is.null(x_num)) {
      x_num <- self$numericalEmbedding(x_num)
      x <- torch::torch_cat(list(x, x_num), dim=2L)
    } else {
      x <- x
    }
    x <- self$classToken(x)
    paddingMask <- torch::torch_zeros(x_cat$size()[1], x_cat$size()[2], 
                                      device = x$device, dtype=torch::torch_bool())
    paddingMask[x_cat==0] <- 1
    paddingMask <- torch::torch_cat(list(paddingMask, 
                                         torch::torch_zeros(x_cat$size()[1], 
                                                            x$size()[2] - x_cat$size()[2],
                                                            device=x$device,
                                                            dtype=torch::torch_bool())), dim=2)
    for (i in 1:length(self$layers)) {
      layer <- self$layers[[i]]
      xResidual <- self$startResidual(layer, 'attention', x)

      if (i==length(self$layers)) {
        dims <- xResidual$shape
        # in final layer take only attention on CLS token
        xResidual <- layer$attention(xResidual[,-1]$view(c(dims[1], 1, dims[3]))$transpose(1,2), 
                                     xResidual$transpose(1,2), 
                                     xResidual$transpose(1,2), paddingMask)
        attnWeights <- xResidual[[2]]
        xResidual <- xResidual[[1]]
        x <- x[,-1]$view(c(dims[1], 1, dims[3]))
        } else {
        # attention input is seq_length x batch_size x embedding_dim
        xResidual <- layer$attention(xResidual$transpose(1,2), 
                                     xResidual$transpose(1,2), 
                                     xResidual$transpose(1,2),
                                     paddingMask,
                                     )[[1]]
        }
      x <- self$endResidual(layer, 'attention', x, xResidual$transpose(1,2))

      xResidual <- self$startResidual(layer, 'ffn', x)
      xResidual <- layer$ffn(xResidual)
      x <- self$endResidual(layer, 'ffn', x, xResidual)
    }
    x <- self$head(x)[,1] # remove singleton dimension
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
    xResidual <-layer[[dropoutKey]](xResidual)
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
    self$embedding <- torch::nn_embedding(numEmbeddings, embeddingDim, padding_idx = 1)
  },
  forward = function(x_cat) {
    x <- self$embedding(x_cat + 1L) # padding idx is 1L
    }
)

numericalEmbedding <- torch::nn_module(
  name='numericalEmbedding',
  initialize = function(numEmbeddings, embeddingDim, bias=FALSE) {
    self$weight <- torch::nn_parameter(torch::torch_empty(numEmbeddings,embeddingDim))
    if (bias) {
    self$bias <- torch::nn_parameter(torch::torch_empty(numEmbeddings, embeddingDim)) 
    } else {
      self$bias <- NULL
    }
    
    for (parameter in list(self$weight, self$bias)) {
      if (!is.null(parameter)) {
        torch::nn_init_kaiming_uniform_(parameter, a=sqrt(5)) 
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
  name='ClassToken',
  initialize = function(dimToken) {
    self$weight <- torch::nn_parameter(torch::torch_empty(dimToken,1))
    torch::nn_init_kaiming_uniform_(self$weight, a=sqrt(5))
  },
  expand = function(dims) {
    newDims <- vector("integer", length(dims) - 1) + 1
    return (self$weight$view(c(newDims,-1))$expand(c(dims, -1)))
    
  },
  forward = function(x) {
    return(torch::torch_cat(c(x, self$expand(c(dim(x)[[1]], 1))), dim=2))
  }
)
