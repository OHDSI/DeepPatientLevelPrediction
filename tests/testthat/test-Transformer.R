settings <- setTransformer(
  numBlocks = 1, dimToken = 8, dimOut = 1,
  numHeads = 2, attDropout = 0.0, ffnDropout = 0.2,
  resDropout = 0.0, dimHidden = 32, 
  estimatorSettings = setEstimator(learningRate = 3e-4,
                                   batchSize=64,
                                   epochs=1),
  randomSample = 1
)

test_that("Transformer settings work", {
  testthat::expect_s3_class(object = settings, class = "modelSettings")
  testthat::expect_equal(settings$fitFunction, "fitEstimator")
  testthat::expect_true(length(settings$param) > 0)
  testthat::expect_error(setTransformer(
    numBlocks = 1, dimToken = 50,
    numHeads = 7
  ))
  testthat::expect_error(setTransformer(
    numBlocks = 1, dimToken = c(2, 4),
    numHeads = c(2, 4)
  ))
  testthat::expect_error(setTransformer(
    numBlocks = 1, dimToken = c(4, 6),
    numHeads = c(2, 4)
  ))
})

test_that("fitEstimator with Transformer works", {
  results <- fitEstimator(trainData$Train, settings, analysisId = 1, analysisPath = testLoc)

  expect_equal(class(results), "plpModel")
  expect_equal(attr(results, "modelType"), "binary")
  expect_equal(attr(results, "saveType"), "file")

  # check prediction between 0 and 1
  expect_gt(min(results$prediction$value), 0)
  expect_lt(max(results$prediction$value), 1)
})

test_that("transformer nn-module works", {
  model <- Transformer(
    catFeatures = 5, numFeatures = 1, numBlocks = 2,
    dimToken = 16, numHeads = 2, attDropout = 0, ffnDropout = 0,
    resDropout = 0, dimHidden = 32
  )

  pars <- sum(sapply(model$parameters, function(x) prod(x$shape)))

  # expected number of parameters
  expect_equal(pars, 5697)

  input <- list()
  input$cat <- torch::torch_randint(0, 5, c(10, 5), dtype = torch::torch_long())
  input$num <- torch::torch_randn(10, 1, dtype = torch::torch_float32())

  output <- model(input)

  # output is correct shape, size of batch
  expect_equal(output$shape, 10)

  input$num <- NULL

  model <- Transformer(
    catFeatures = 5, numFeatures = 0, numBlocks = 2,
    dimToken = 16, numHeads = 2, attDropout = 0, ffnDropout = 0,
    resDropout = 0, dimHidden = 32
  )
  output <- model(input)
  expect_equal(output$shape, 10)
})

test_that("Default Transformer works", {
  defaultTransformer <- setDefaultTransformer()
  params <- defaultTransformer$param[[1]]
  
  expect_equal(params$numBlocks, 3)
  expect_equal(params$dimToken, 192)
  expect_equal(params$numHeads, 8)
  expect_equal(params$resDropout, 0.0)
  expect_equal(params$attDropout, 0.2)
  
  settings <- attr(defaultTransformer, 'settings')
  
  expect_equal(settings$name, 'defaultTransformer')
}) 

test_that("Errors are produced by settings function", {
  randomSample <- 2
  
  expect_error(setTransformer(randomSample = randomSample))
})
