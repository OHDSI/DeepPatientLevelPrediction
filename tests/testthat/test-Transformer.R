settings <- setTransformer(numBlocks=1, dimToken=8, dimOut = 1,
                           numHeads = 2, attDropout = 0.0, ffnDropout = 0.2,
                           resDropout = 0.0,dimHidden = 32, batchSize = 64,
                           epochs = 1, randomSamples = 1)

test_that('Transformer settings work', {
  
  testthat::expect_s3_class(object = settings, class = 'modelSettings')
  testthat::expect_equal(settings$fitFunction, 'fitEstimator') 
  testthat::expect_true(length(settings$param) > 0 )
})

test_that('fitEstimator with Transformer works', {
  
  results <- fitEstimator(trainData$Train, settings$param, analysisId=1)
  
  expect_equal(class(results), 'plpModel')
  expect_equal(attr(results, 'modelType'), 'binary')
  expect_equal(attr(results, 'saveType'), 'file')
  
  # check prediction between 0 and 1
  expect_gt(min(results$prediction$value), 0)
  expect_lt(max(results$prediction$value), 1)
  
})

test_that('transformer nn-module works', {
  model <- Transformer(catFeatures=5, numFeatures=1, numBlocks=2, 
                      dimToken=16, numHeads=2, attDropout=0, ffnDropout=0,
                      resDropout=0, dimHidden=32)
  
  pars <- sum(sapply(model$parameters, function(x) prod(x$shape)))
  
  # expected number of parameters
  expect_equal(pars, 5697)
  
  input <- list()
  input$cat <- torch::torch_randint(0, 5, c(10,5), dtype = torch::torch_long())
  input$num <- torch::torch_randn(10,1, dtype = torch::torch_float32())
  
  
  output <- model(input)
  
  # output is correct shape, size of batch
  expect_equal(output$shape, 10)
  
  input$num <- NULL
  
  model <- Transformer(catFeatures=5, numFeatures=0, numBlocks=2, 
                       dimToken=16, numHeads=2, attDropout=0, ffnDropout=0,
                       resDropout=0, dimHidden=32)
  output <- model(input)
  expect_equal(output$shape, 10)
})