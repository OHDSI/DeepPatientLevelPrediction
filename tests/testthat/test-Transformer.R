settings <- setTransformer(
  numBlocks = 1,
  dimToken = 8,
  dimOut = 1,
  numHeads = 2,
  attDropout = 0.0,
  ffnDropout = 0.2,
  dimHidden = 32,
  estimatorSettings = setEstimator(
    learningRate = 3e-4,
    batchSize = 64,
    epochs = 1
  ),
  randomSample = 1
)

test_that("Transformer settings work", {
  expect_s3_class(object = settings, class = "modelSettings")
  expect_equal(settings$fitFunction, "DeepPatientLevelPrediction::fitEstimator")
  expect_true(length(settings$param) > 0)
  expect_error(setTransformer(
    numBlocks = 1, dimToken = 50,
    numHeads = 7
  ))
  expect_error(setTransformer(
    numBlocks = 1, dimToken = c(2, 4),
    numHeads = c(2, 4)
  ))
  expect_error(setTransformer(
    numBlocks = 1, dimToken = c(4, 6),
    numHeads = c(2, 4)
  ))
  expect_error(setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      maxSequenceLength = "notMaxSequenceLength",
      truncation = "tail",
      time_tokens = TRUE
    )
  ))
  expect_error(setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      maxSequenceLength = 256,
      truncation = "max",
      time_tokens = TRUE
    )
  ))

  transformerSettings <- setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      maxSequenceLength = "max",
      truncation = "tail",
      time_tokens = TRUE
    )
  )
  expect_true(attr(transformerSettings$param, "temporalModel"))
  temporalSettings <- attr(transformerSettings$param, "temporalSettings")
  expect_equal(temporalSettings$maxSequenceLength, "max")
  expect_equal(temporalSettings$truncation, "tail")
  expect_equal(temporalSettings$time_tokens, TRUE)
})

test_that("fitEstimator with Transformer works", {
  results <-
    fitEstimator(trainData$Train,
      settings,
      analysisId = 1,
      analysisPath = testLoc
    )

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
    feature_info = dataset$get_feature_info(),
    num_blocks = 2,
    dim_token = 16,
    num_heads = 2,
    att_dropout = 0,
    ffn_dropout = 0,
    dim_hidden = 32
  )

  pars <- sum(reticulate::iterate(model$parameters(), function(x) x$numel()))

  # expected number of parameters
  expect_equal(pars, 7313)

  input <- list()
  input$feature_ids <- torch$randint(0L, 5L, c(10L, 5L), dtype = torch$long)
  input$feature_values <- torch$randn(10L, 5L, dtype = torch$float32)

  output <- model(input)

  # output is correct shape, size of batch
  expect_equal(output$shape[0], 10L)

  model <- transformer(
    feature_info = dataset$get_feature_info(),
    num_blocks = 2,
    dim_token = 16,
    num_heads = 2,
    att_dropout = 0,
    ffn_dropout = 0,
    dim_hidden = 32
  )
  output <- model(input)
  expect_equal(output$shape[0], 10L)
})

test_that("Default Transformer works", {
  defaultTransformer <- setDefaultTransformer()
  params <- defaultTransformer$param[[1]]

  expect_equal(params$numBlocks, 3)
  expect_equal(params$dimToken, 192)
  expect_equal(params$numHeads, 8)
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
  modelSettings <- setTransformer(
    dimToken = dimToken,
    dimHiddenRatio = dimHiddenRatio,
    dimHidden = NULL,
    randomSample = randomSample
  )
  dimHidden <- unlist(lapply(modelSettings$param, function(x) x$dimHidden))
  tokens <- unlist(lapply(modelSettings$param, function(x) x$dimToken))
  testthat::expect_true(all(dimHidden == dimHiddenRatio * tokens))
  testthat::expect_error(setTransformer(
    dimHidden = NULL,
    dimHiddenRatio = NULL
  ))
  testthat::expect_error(setTransformer(
    dimHidden = 256,
    dimHiddenRatio = 4 / 3
  ))
})

test_that("numerical embedding works as expected", {
  embeddings <- 32L # size of embeddings
  features <- 2L # number of numerical features
  patients <- 9L

  featureIds <- torch$cat(c(
    torch$ones(9L, 1L, dtype = torch$long),
    torch$ones(9L, 1L, dtype = torch$long) * 2L
  ), dim = 1L)
  values <- torch$randn(c(patients, features))

  numericalEmbeddingClass <-
    reticulate::import_from_path("Embeddings", path = path)$NumericalEmbedding
  numericalEmbedding <- numericalEmbeddingClass(
    num_embeddings = features,
    embedding_dim = embeddings,
    bias = TRUE
  )
  out <- numericalEmbedding(featureIds, values)

  # should be patients x features x embedding size
  expect_equal(out$shape[[0]], patients)
  expect_equal(out$shape[[1]], features)
  expect_equal(out$shape[[2]], embeddings)

  numericalEmbedding <- numericalEmbeddingClass(
    num_embeddings = features,
    embedding_dim = embeddings,
    bias = FALSE
  )

  out <- numericalEmbedding(featureIds, values)
  expect_equal(out$shape[[0]], patients)
  expect_equal(out$shape[[1]], features)
  expect_equal(out$shape[[2]], embeddings)
})

test_that("temporal transformer works", {
  temporalCovSettings <- FeatureExtraction::createTemporalSequenceCovariateSettings(
    useDemographicsAge = TRUE,
    useDemographicsGender = TRUE,
    useConditionOccurrence = TRUE,
    sequenceEndDay = -65,
    sequenceStartDay = -1
  )
  plpData <- PatientLevelPrediction::getPlpData(
    databaseDetails = databaseDetails,
    restrictPlpDataSettings = restrictPlpDataSettings,
    covariateSettings = temporalCovSettings
  )
  trainData <- PatientLevelPrediction::splitData(
    plpData = plpData,
    population = population,
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42)
  )

  settings <- setTransformer(
    numBlocks = 1,
    dimToken = 8,
    dimOut = 1,
    numHeads = 2,
    attDropout = 0.0,
    ffnDropout = 0.2,
    dimHidden = 32,
    temporal = TRUE,
    temporalSettings = list(
      maxSequenceLength = 16,
      truncation = "tail",
      timeTokens = FALSE
    ),
    estimatorSettings = setEstimator(
      learningRate = 3e-4,
      batchSize = 64,
      epochs = 1
    ),
    randomSample = 1
  )
  results <-
    fitEstimator(trainData$Train,
      settings,
      analysisId = "temporalTransformer",
      analysisPath = testLoc
    )

  expect_equal(class(results), "plpModel")
  expect_equal(attr(results, "modelType"), "binary")
  expect_equal(attr(results, "saveType"), "file")

  # check prediction between 0 and 1
  expect_gt(min(results$prediction$value), 0)
  expect_lt(max(results$prediction$value), 1)
})

# below tests work with PLP version 6.5.0.9999 (Develop), 
# test_that("Positional encodings work", {
#   # if temporal is FALSE, positional encodings should not be set and accounted
#   # for in the hyperparameter search
#   results <- setTransformer(
#     temporal = FALSE,
#     temporalSettings = list(
#       positionalEncoding = list(list(name = "A"), list(name = "B"))
#     ),
#     hyperParamSearch = "grid"
#   )
#
#   expect_equal(length(results$param), 1)
#   expect_false("positionalEncoding" %in% results$modelParamNames)
#
#
#   results <- setTransformer(
#     temporal = TRUE,
#     temporalSettings = list(
#       positionalEncoding = NULL,
#       maxSequenceLength = 256
#     ),
#     hyperParamSearch = "grid"
#   )
#   # if temporal is TRUE and positionalEncoding is NULL, it should not be set
#   # and accounted for in the hyperparameter search
#   expect_equal(length(results$param), 1)
#   expect_false("positionalEncoding" %in% results$modelParamNames)
#   expect_false("positionalEncoding" %in% names(results$param[[1]]))
#
#   
#
#   results <- setTransformer(
#     temporal = TRUE,
#     temporalSettings = list(
#       positionalEncoding = "SinusoidalPE",
#       maxSequenceLength = 256
#     )
#   )
#   
#   expect_equal(length(results$param), 1)
#   expect_true("positionalEncoding" %in% results$modelParamNames)
#   expect_equal(results$param[[1]]$positionalEncoding, list(name = "SinusoidalPE"))
#   
#   # Handles a single PE provided as a list
#   peConfig <- list(name = "LearnablePE", dropout = 0.25)
#   results <- setTransformer(
#     temporal = TRUE,
#     temporalSettings = list(
#       positionalEncoding = peConfig,
#       maxSequenceLength = 256
#     ),
#     hyperParamSearch = "grid"
#   )
#   
#   expect_equal(length(results$param), 1)
#   expect_true("positionalEncoding" %in% results$modelParamNames)
#   expect_equal(results$param[[1]]$positionalEncoding, peConfig)
#
#   # Handles multiple PEs provided
#   peConfigs <- list(
#     list(name = "SinusoidalPE", dropout = 0.1),
#     list(name = "LearnablePE", dropout = 0.2)
#   )
#   
#   results <- setTransformer(
#     temporal = TRUE,
#     temporalSettings = list(
#       positionalEncoding = peConfigs,
#       maxSequenceLength = 256
#     ),
#     hyperParamSearch = "grid"
#   )
#   
#   expect_equal(length(results$param), 2)
#   expect_true("positionalEncoding" %in% results$modelParamNames)
#   expect_equal(results$param[[1]]$positionalEncoding, peConfigs[[1]])
#   expect_equal(results$param[[2]]$positionalEncoding, peConfigs[[2]])
#   
#
#   # This tests searching over parameters *within* one PE type.
#   peConfig <- list(name = "SinusoidalPE", dropout = c(0.1, 0.2))
#   
#   results <- setTransformer(
#     temporal = TRUE,
#     temporalSettings = list(
#       positionalEncoding = peConfig,
#       maxSequenceLength = 256
#     ),
#     hyperParamSearch = "grid"
#   )
#   
#   expect_equal(length(results$param), 2)
#   expect_true("positionalEncoding" %in% results$modelParamNames)
#   
#   expect_equal(results$param[[1]]$positionalEncoding, list(name = "SinusoidalPE", dropout = 0.1))
#   expect_equal(results$param[[2]]$positionalEncoding, list(name = "SinusoidalPE", dropout = 0.2))
#
#   # This tests searching over multiple PEs with parameter values.
#   peConfig <- list(
#     list(name = "SinusoidalPE", dropout = c(0.0, 0.1)),
#     list(name = "LearnablePE", dropout = 0.15)
#   )
#   
#   results <- setTransformer(
#     dimToken = c(64, 128),
#     temporal = TRUE,
#     temporalSettings = list(
#       positionalEncoding = peConfig,
#       maxSequenceLength = 256
#     ),
#     hyperParamSearch = "grid"
#   )
#   
#   # Expected number of combinations = 2 (for dimToken) * (2 + 1) (for PEs) = 6
#   expect_equal(length(results$param), 6)
#   expect_true("positionalEncoding" %in% results$modelParamNames)
#   
#   # Spot-check some of the combinations to ensure correctness
#   paramSummary <- sapply(results$param, function(p) {
#     paste0("dimToken=", p$dimToken, ", PE_name=", p$positionalEncoding$name, ", dropout=", p$positionalEncoding$dropout)
#   })
#
#   # Expected combinations
#   expectedSummary <- c(
#     "dimToken=64, PE_name=SinusoidalPE, dropout=0",
#     "dimToken=128, PE_name=SinusoidalPE, dropout=0",
#     "dimToken=64, PE_name=SinusoidalPE, dropout=0.1",
#     "dimToken=128, PE_name=SinusoidalPE, dropout=0.1",
#     "dimToken=64, PE_name=LearnablePE, dropout=0.15",
#     "dimToken=128, PE_name=LearnablePE, dropout=0.15"
#   )
#   
#   # Use sort to make the comparison order-independent
#   expect_equal(sort(paramSummary), sort(expectedSummary))
# })

test_that("setTransformer errors early on Flash when environment is not compatible", {
  expect_error(
    setTransformer(
      numBlocks = 1, dimToken = 192, numHeads = 8, dimHidden = 256,
      attnImplementation = "flash",
      estimatorSettings = setEstimator(device = "cpu"),
      temporal = FALSE
    ),
    regexp = "FlashAttention-2 environment validation failed",
    fixed = FALSE
  )
})

test_that("setTransformer errors on invalid attnImplementation", {
  expect_error(
    DeepPatientLevelPrediction::setTransformer(
      numBlocks = 1, dimToken = 192, numHeads = 8, dimHidden = 256,
      attnImplementation = "flashy",      # invalid
      temporal = FALSE
    ),
    regexp = "attnImplementation must be either 'sdpa' or 'flash'. You provided:",
    fixed = FALSE
  )
})

test_that("setTransformer converts character positionalEncoding to list(name=...)", {
  res <- DeepPatientLevelPrediction::setTransformer(
    numBlocks = 1, dimToken = 192, numHeads = 8, dimHidden = 256,
    attnImplementation = "sdpa",         
    temporal = TRUE,
    temporalSettings = list(
      positionalEncoding = "SinusoidalPE",
      maxSequenceLength = 32,
      truncation = "tail",
      timeTokens = FALSE
    ),
    estimatorSettings = setEstimator(device = "cpu"),
    hyperParamSearch = "random",
    randomSample = 1
  )

  ts <- attr(res$param, "temporalSettings")
  expect_true(is.list(ts$positionalEncoding))
  expect_equal(ts$positionalEncoding$name, "SinusoidalPE")

  expect_true("positionalEncoding" %in% res$modelParamNames)
})

test_that("setTransformer aggregates flashParamCheck errors into a single message", {
  fake <- makeFakeTorch(
    cudaAvailable = TRUE,
    cudaVersion = "12.1",
    capMajor = 8L, capMinor = 0L,
    deviceName = "NVIDIA A100-SXM4-40GB",
    bf16Supported = TRUE
  )
  localMockFlashBindings(fakeTorch = fake, flashModuleAvailable = TRUE, .scope = environment())

  expect_error(
    DeepPatientLevelPrediction::setTransformer(
      numBlocks = 1, dimToken = 192, numHeads = 8, dimHidden = 256,
      attnImplementation = "flash",
      temporal = TRUE,
      temporalSettings = list(
        positionalEncoding = "RelativePE", 
        maxSequenceLength = 32,
        truncation = "tail",
        timeTokens = FALSE
      ),
      estimatorSettings = setEstimator(device = "cuda", precision = "bfloat16"),
      hyperParamSearch = "random",
      randomSample = 1
    ),
    regexp = "FlashAttention-2 is not supported for the following hyperparameter combinations:",
    fixed = FALSE
  )
})
