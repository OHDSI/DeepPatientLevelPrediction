settings <- setTransformer(
  numBlocks = 1L, 
  dimToken = 8L, 
  dimOut = 1L,
  numHeads = 2L, 
  attDropout = 0.0, 
  ffnDropout = 0.2,
  resDropout = 0.0, 
  dimHidden = 32L, 
  estimatorSettings = setEstimator(learningRate = 3e-4,
                                   batchSize=64L,
                                   epochs=1L),
  randomSample = 1
)

test_that("Transformer settings work", {
  testthat::expect_s3_class(object = settings, class = "modelSettings")
  testthat::expect_equal(settings$fitFunction, "fitEstimator")
  testthat::expect_true(length(settings$param) > 0)
  testthat::expect_error(setTransformer(
    numBlocks = 1L, dimToken = 50L,
    numHeads = 7L
  ))
  testthat::expect_error(setTransformer(
    numBlocks = 1L, dimToken = c(2L, 4L),
    numHeads = c(2L, 4L)
  ))
  testthat::expect_error(setTransformer(
    numBlocks = 1L, dimToken = c(4L, 6L),
    numHeads = c(2L, 4L)
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
  Transformer <- reticulate::import_from_path("Transformer", path=path)$Transformer
  model <- Transformer(
    cat_features = 5L, 
    num_features = 1L, 
    num_blocks = 2L,
    dim_token = 16L, 
    num_heads = 2L, 
    att_dropout = 0, 
    ffn_dropout = 0,
    res_dropout = 0, 
    dim_hidden = 32L
  )

  pars <- sum(reticulate::iterate(model$parameters(), function(x) x$numel()))

  # expected number of parameters
  expect_equal(pars, 5697)

  input <- list()
  input$cat <- torch$randint(0L, 5L, c(10L, 5L), dtype = torch$long)
  input$num <- torch$randn(10L, 1L, dtype = torch$float32)

  output <- model(input)

  # output is correct shape, size of batch
  expect_equal(output$shape[0], 10L)

  input$num <- NULL

  model <- Transformer(
    cat_features = 5L, 
    num_features = 0L, 
    num_blocks = 2L,
    dim_token = 16L, 
    num_heads = 2L, 
    att_dropout = 0, 
    ffn_dropout = 0,
    res_dropout = 0, 
    dim_hidden = 32L
  )
  output <- model(input)
  expect_equal(output$shape[0], 10L)
})

test_that("Default Transformer works", {
  defaultTransformer <- setDefaultTransformer()
  params <- defaultTransformer$param[[1]]
  
  expect_equal(params$numBlocks, 3L)
  expect_equal(params$dimToken, 192L)
  expect_equal(params$numHeads, 8L)
  expect_equal(params$resDropout, 0.0)
  expect_equal(params$attDropout, 0.2)
  
  settings <- attr(defaultTransformer, 'settings')
  
  expect_equal(settings$name, 'defaultTransformer')
}) 

test_that("Errors are produced by settings function", {
  randomSample <- 2
  
  expect_error(setTransformer(randomSample = randomSample))
})

test_that("dimHidden ratio works as expected", {
  randomSample <- 4
  dimToken <- c(64L, 128L, 256L, 512L)
  dimHiddenRatio <- 2L
  modelSettings <- setTransformer(dimToken = dimToken,
                                  dimHiddenRatio = dimHiddenRatio,
                                  dimHidden = NULL,
                                  randomSample = randomSample)
  dimHidden <- unlist(lapply(modelSettings$param, function(x) x$dimHidden))
  tokens <-   unlist(lapply(modelSettings$param, function(x) x$dimToken))
  expect_true(all(dimHidden == dimHiddenRatio * tokens))

})
