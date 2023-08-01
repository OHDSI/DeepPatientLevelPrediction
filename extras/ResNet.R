ResNet <- torch::nn_module(
  name='ResNet',
  initialize=function(catFeatures, numFeatures=0, sizeEmbedding, sizeHidden, numLayers,
                      hiddenFactor, activation=torch::nn_relu, 
                      normalization=torch::nn_batch_norm1d, hiddenDropout=NULL,
                      residualDropout=NULL, d_out=1) {
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
  
  forward=function(x_cat) {
    x_cat <- self$embedding(x_cat + 1L)
    x_num <- NULL
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