singleLayerNN <- function(inputN, layer1, outputN = 2, layer_dropout){
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


doubleLayerNN <- function(inputN, layer1, 
                          layer2, outputN,
                          layer_dropout){
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


tripleLayerNN <- function(inputN, layer1, 
                          layer2, layer3,
                          outputN, layer_dropout){
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