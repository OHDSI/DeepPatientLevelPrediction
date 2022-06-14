#' A single layer neural network
#' @param inputN        Input neurons
#' @param layer1        Layer 1 neurons
#' @param outputN       Output neurons
#' @param layer_dropout Layer dropout to use
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

#' Double layer neural network
#' @param inputN        Input neurons
#' @param layer1        Layer 1 neurons
#' @param layer2        Layer 2 neurons
#' @param outputN       output neurons
#' @param layer_dropout layer_dropout to use
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

#' Triple layer neural network
#' @param inputN        Input neurons
#' @param layer1        amount of layer 1 neurons
#' @param layer2        amount of layer 2 neurons
#' @param layer3        amount of layer 3 neurons
#' @param outputN       Number of output neurons
#' @param layer_dropout The dropout to use in layer
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