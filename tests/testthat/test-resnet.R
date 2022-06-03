test_that("resnet nn-module works ", {
 
  
  model <- ResNet(catFeatures=5, numFeatures=1, sizeEmbedding=5, 
                  sizeHidden=16, numLayers=1, hiddenFactor=2,
                  activation=torch::nn_relu, 
                  normalization=torch::nn_batch_norm1d, hiddenDropout=0.3,
                  residualDropout=0.3, d_out=1)
  
  pars <- sum(sapply(model$parameters, function(x) prod(x$shape)))
  
  # expected number of parameters
  expect_equal(pars, 1295)
  
  input <- list()
  input$cat <- torch::torch_randint(0, 5, c(10,5), dtype = torch::torch_long())
  input$num <- torch::torch_randn(10,1, dtype = torch::torch_float32())
  
  
  output <- model(input)
  
  # output is correct shape
  expect_equal(output$shape, 10)
  
  input$num <- NULL
  model <- ResNet(catFeatures=5, numFeatures=0, sizeEmbedding=5, 
                  sizeHidden=16, numLayers=1, hiddenFactor=2,
                  activation=torch::nn_relu, 
                  normalization=torch::nn_batch_norm1d, hiddenDropout=0.3,
                  residualDropout=0.3, d_out=1)
  output <- model(input)
  # model works without numeric variables
  expect_equal(output$shape, 10)
  
}
)

test_that('setResNet settings object works', {

  modelSettings <- setResNet(numLayers = 1, sizeHidden = 1, hiddenFactor = 1,
              residualDropout = 0.1, hiddenDropout = 0.1, sizeEmbedding = 12,
              learningRate = 3e-4, hyperParamSearch = 'random', randomSample = 1)
  
  expect_equal(length(modelSettings$param), 1)
  
})
  