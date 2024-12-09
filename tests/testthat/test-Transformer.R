settings <- setTransformer(
  numBlocks = 1,
  dimToken = 8,
  dimOut = 1,
  numHeads = 2,
  attDropout = 0.0,
  ffnDropout = 0.2,
  resDropout = 0.0,
  dimHidden = 32,
  estimatorSettings = setEstimator(learningRate = 3e-4,
                                   batchSize = 64,
                                   epochs = 1),
  randomSample = 1
)

test_that("Transformer settings work", {
  testthat::expect_s3_class(object = settings, class = "modelSettings")
  testthat::expect_equal(settings$fitFunction, "DeepPatientLevelPrediction::fitEstimator")
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
  results <-
    fitEstimator(trainData$Train,
                 settings,
                 analysisId = 1,
                 analysisPath = testLoc)

  expect_equal(class(results), "plpModel")
  expect_equal(attr(results, "modelType"), "binary")
  expect_equal(attr(results, "saveType"), "file")

  # check prediction between 0 and 1
  expect_gt(min(results$prediction$value), 0)
  expect_lt(max(results$prediction$value), 1)
})

test_that("transformer nn-module works", {
  transformer <-
    reticulate::import_from_path("Transformer", path = path)$Transformer
  model <- transformer(
    feature_info = list("categorical_features" = 5L,
                        "numerical_features" = 1L),
    num_blocks = 2,
    dim_token = 16,
    num_heads = 2,
    att_dropout = 0,
    ffn_dropout = 0,
    res_dropout = 0,
    dim_hidden = 32
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

  model <- transformer(
    feature_info = list(categorical_features = 5),
    num_blocks = 2,
    dim_token = 16,
    num_heads = 2,
    att_dropout = 0,
    ffn_dropout = 0,
    res_dropout = 0,
    dim_hidden = 32
  )
  output <- model(input)
  expect_equal(output$shape[0], 10L)  
  input$num <- reticulate::py_none()
  output <- model(input)
  expect_equal(output$shape[0], 10L)
  input$num <- reticulate::py_none()
  output <- model(input)
  expect_equal(output$shape[0], 10L)
})

test_that("Default Transformer works", {
  defaultTransformer <- setDefaultTransformer()
  params <- defaultTransformer$param[[1]]

  expect_equal(params$numBlocks, 3)
  expect_equal(params$dimToken, 192)
  expect_equal(params$numHeads, 8)
  expect_equal(params$resDropout, 0.0)
  expect_equal(params$attDropout, 0.2)

  settings <- attr(defaultTransformer, "settings")

  expect_equal(settings$name, "defaultTransformer")
})

test_that("Errors are produced by settings function", {
  randomSample <- 2

  expect_error(setTransformer(randomSample = randomSample))
})

test_that("dimHidden ratio works as expected", {
  randomSample <- 4
  dimToken <- c(64, 128, 256, 512)
  dimHiddenRatio <- 2
  modelSettings <- setTransformer(dimToken = dimToken,
                                  dimHiddenRatio = dimHiddenRatio,
                                  dimHidden = NULL,
                                  randomSample = randomSample)
  dimHidden <- unlist(lapply(modelSettings$param, function(x) x$dimHidden))
  tokens <-   unlist(lapply(modelSettings$param, function(x) x$dimToken))
  testthat::expect_true(all(dimHidden == dimHiddenRatio * tokens))
  testthat::expect_error(setTransformer(dimHidden = NULL,
                                        dimHiddenRatio = NULL))
  testthat::expect_error(setTransformer(dimHidden = 256,
                                        dimHiddenRatio = 4 / 3))

})

test_that("numerical embedding works as expected", {
  embeddings <- 32L # size of embeddings
  features <- 2L # number of numerical features
  patients <- 9L

  numTensor <- torch$randn(c(patients, features))

  numericalEmbeddingClass <-
    reticulate::import_from_path("ResNet", path = path)$NumericalEmbedding
  numericalEmbedding <- numericalEmbeddingClass(num_embeddings = features,
                                                embedding_dim = embeddings,
                                                bias = TRUE)
  out <- numericalEmbedding(numTensor)

  # should be patients x features x embedding size
  expect_equal(out$shape[[0]], patients)
  expect_equal(out$shape[[1]], features)
  expect_equal(out$shape[[2]], embeddings)

  numericalEmbedding <- numericalEmbeddingClass(num_embeddings = features,
                                                embedding_dim = embeddings,
                                                bias = FALSE)

  out <- numericalEmbedding(numTensor)
  expect_equal(out$shape[[0]], patients)
  expect_equal(out$shape[[1]], features)
  expect_equal(out$shape[[2]], embeddings)

})

test_that("numerical embedding works as expected", {
  embeddings <- 32L # size of embeddings
  features <- 2L # number of numerical features
  patients <- 9L 
  
  numTensor <- torch$randn(c(patients, features))
  
  numericalEmbeddingClass <- reticulate::import_from_path("ResNet", path=path)$NumericalEmbedding
  numericalEmbedding <- numericalEmbeddingClass(num_embeddings = features,
                                                embedding_dim = embeddings,
                                                bias = TRUE)
  out <- numericalEmbedding(numTensor)
  
  # should be patients x features x embedding size
  expect_equal(out$shape[[0]], patients)
  expect_equal(out$shape[[1]], features)
  expect_equal(out$shape[[2]], embeddings)
  
  numericalEmbedding <- numericalEmbeddingClass(num_embeddings = features,
                                                embedding_dim = embeddings,
                                                bias = FALSE)
  
  out <- numericalEmbedding(numTensor)
  expect_equal(out$shape[[0]], patients)
  expect_equal(out$shape[[1]], features)
  expect_equal(out$shape[[2]], embeddings)
  
  })
