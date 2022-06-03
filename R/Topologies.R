#' @export
singleLayerNN <- function(inputN, layer1, outputN = 2, layer_dropout){
  
  self <- NA # fixing R check
  
  net <- torch::nn_module(
    "classic_net",
    
    initialize = function(){
      self$linear1 = torch::nn_linear(inputN, layer1)
      self$linear2 = torch::nn_linear(layer1, outputN)
      self$softmax = torch::nn_softmax(outputN)
    },
    
    forward = function(x){
      x %>%
        self$linear1() %>%
        torch::nnf_dropout(p = layer_dropout) %>%
        self$linear2() %>%
        torch::nnf_dropout(p = layer_dropout) %>%
        self$softmax()
    }
  )
  return(net())
}

#' @export
doubleLayerNN <- function(inputN, layer1, 
                          layer2, outputN,
                          layer_dropout){
  
  self <- NA # fixing R check
  
  net <- torch::nn_module(
    "classic_net",
    
    initialize = function(){
      self$linear1 = torch::nn_linear(inputN, layer1)
      self$linear2 = torch::nn_linear(layer1, layer2)
      self$linear3 = torch::nn_linear(layer2, outputN)
      self$softmax = torch::nn_softmax(outputN)
    },
    
    forward = function(x){
      x %>%
        self$linear1() %>%
        torch::nnf_dropout(p = layer_dropout) %>%
        self$linear2() %>%
        torch::nnf_dropout(p = layer_dropout) %>%
        self$linear3() %>%
        self$softmax()
    }
  )
  return(net())
}

#' @export
tripleLayerNN <- function(inputN, layer1, 
                          layer2, layer3,
                          outputN, layer_dropout){
  
  self <- NA # fixing R check
  
  net <- torch::nn_module(
    "classic_net",
    
    initialize = function(){
      self$linear1 = torch::nn_linear(inputN, layer1)
      self$linear2 = torch::nn_linear(layer1, layer2)
      self$linear3 = torch::nn_linear(layer2, layer3)
      self$linear4 = torch::nn_linear(layer3, outputN)
      self$softmax = torch::nn_softmax(outputN)
    },
    
    forward = function(x){
      x %>%
        self$linear1() %>%
        torch::nnf_dropout(p = layer_dropout) %>%
        self$linear2() %>%
        torch::nnf_dropout(p = layer_dropout) %>%
        self$linear3() %>%
        torch::nnf_dropout(p = layer_dropout) %>%
        self$linear4() %>%
        self$softmax()
      
    }
  )
  model <- net()
}

# Multi-resolution CNN1 model 
# Stucture based on https://arxiv.org/pdf/1608.00647.pdf CNN1

MRCovNN_submodel1 <- function(kernel_size){
  
  self <- NA # adding this to stop R check warning
  
  net <- torch::nn_module(
    "MRCovNN_submodel1",
    
    initialize = function(){
      self$conv1 <- torch::nn_conv2d(in_channel = 1,
                                     out_channel = 1,
                                     kernel_size = kernel_size)
      
    },
    
    forward = function(x){
      x %>%
        torch::nnf_max_pool2d(kernel_size) %>%
        self$conv1() %>%
        torch::nnf_relu()
    }
  )
  return(net())
}

MRCovNN_submodel2 <- function(kernel_size){
  
  self <- NA # adding this to stop R check warning
  
  net <- torch::nn_module(
    "MRCovNN_submodel2",
    
    initialize = function(){
      self$maxPool <- torch::nn_max_pool2d(kernel_size = kernel_size)
      self$conv1 <- torch::nn_conv2d(in_channels = 1,
                                     out_channels = 1,
                                     kernel_size = kernel_size)
    },
    
    forward = function(x){
      x %>%
        self$maxPool() %>%
        self$conv1() %>%
        torch::nnf_relu()
    }
  )
  return(net())
}


MRCovNN_submodel3 <- function(kernel_size){
  
  self <- NA # adding this to stop R check warning
  
  net <- torch::nn_module(
    "MRCovNN_submodel3",
    
    initialize = function(){
      self$maxPool <- torch::nn_max_pool2d(kernel_size = kernel_size)
      self$conv1 <- torch::nn_conv2d(in_channels = 1,
                                     out_channels = 1,
                                     kernel_size = kernel_size)
      
      self$conv2 <- torch::nn_conv2d(in_channels = 1,
                                     out_channels = 1,
                                     kernel_size = kernel_size)
    },
    
    forward = function(x){
      x %>%
        self$conv1() %>%
        torch::nnf_relu() %>%
        self$maxPool() %>%
        self$conv2() %>%
        torch::nnf_relu()
    }
  )
  return(net())
}


# submodel1 = MRCovNN_submodel1(kernelSize = c(4,1))
# submodel1 = submodel1(x)
# modelList = list(submodel1, submodel2, submodel3)

MultiResolutionCovNN <- function(
  modelList = list(
    MRCovNN_submodel1(kernel_size = c(4,1)), 
    MRCovNN_submodel2(kernel_size = c(4,1)),  
    MRCovNN_submodel3(kernel_size = c(4,1))
  ),
  kernelSize,
  dropout,
  inputN,
  layer1,
  layer2,
  outputN = 2
){
  self <- NA # adding this to stop R check warning
  
  net <- torch::nn_module(
    "MultiResolutionCovNN",
    
    initialize = function(){
      self$linear1 <- torch::nn_linear(inputN, layer1)
      self$linear2 <- torch::nn_linear(layer1, layer2)
      self$linear3 <- torch::nn_linear(layer2, outputN)
    },
    
    forward = function(){
      
      torch::torch_cat(modelList, 3) %>%
        torch::nnf_dropout(p = dropout) %>%
        self$linear1() %>%
        torch::nnf_relu() %>%
        torch::nnf_dropout(p = dropout) %>%
        self$linear2() %>%
        torch::nnf_relu() %>%
        torch::nnf_dropout(p = dropout) %>%
        self$linear3() %>%
        torch::nnf_softmax(outputN)
      
    }
  )
  return(net())
}
