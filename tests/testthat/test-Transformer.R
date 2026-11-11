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
  ), "divisible by numHeads", fixed = TRUE)
  expect_error(setTransformer(
    numBlocks = 1, dimToken = c(2, 4),
    numHeads = c(2, 4)
  ), "not for dimToken = 2 and numHeads = 4", fixed = TRUE)
  expect_error(setTransformer(
    numBlocks = 1, dimToken = c(4, 6),
    numHeads = c(2, 4)
  ), "divisible by numHeads", fixed = TRUE)
  expect_error(setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      maxSequenceLength = "notMaxSequenceLength",
      truncation = "tail",
      time_tokens = TRUE
    )
  ), "maxSequenceLength must be either", fixed = TRUE)
  expect_error(setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      maxSequenceLength = 256,
      truncation = "max",
      time_tokens = TRUE
    )
  ), "Only truncation method 'tail' is supported", fixed = TRUE)

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
  skip_if_no_python()
  results <-
    fitEstimator(simTrainData$Train,
      settings,
      analysisId = 1,
      analysisPath = tempdir()
    )

  expect_equal(class(results), "plpModel")
  expect_equal(attr(results, "modelType"), "binary")
  expect_equal(attr(results, "saveType"), "file")

  # check prediction between 0 and 1
  expect_gt(min(results$prediction$value), 0)
  expect_lt(max(results$prediction$value), 1)
})

test_that("transformer nn-module works", {
  skip_if_no_python()
  transformer <- importModule("Transformer")$Transformer
  newModel <- function(featureInfo) {
    transformer(
      feature_info = featureInfo,
      num_blocks = 2,
      dim_token = 16,
      num_heads = 2,
      att_dropout = 0,
      ffn_dropout = 0,
      dim_hidden = 32
    )
  }
  # one numeric and four binary features
  model <- newModel(mappedFeatureInfo(
    covariateIds = 1:5 + 100, conceptIds = 1:5 + 1000,
    isBinary = c("N", "Y", "Y", "Y", "Y")
  ))

  pars <- sum(reticulate::iterate(model$parameters(), function(x) x$numel()))

  # expected number of parameters
  expect_equal(pars, 5713)

  input <- list()
  input$feature_ids <- torch$randint(0L, 5L, c(10L, 5L), dtype = torch$long)
  input$feature_values <- torch$randn(10L, 5L, dtype = torch$float32)

  output <- model(input)

  # output is correct shape, size of batch
  expect_equal(output$shape[0], 10L)

  model <- newModel(mappedFeatureInfo(
    covariateIds = 1:5 + 100, conceptIds = 1:5 + 1000, isBinary = rep("Y", 5)
  ))
  output <- model(input)
  # model works without numeric variables
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

  expect_error(setTransformer(randomSample = randomSample), "Chosen amount of randomSamples is higher than the", fixed = TRUE)
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
  ), "cannot be both set or both NULL", fixed = TRUE)
  testthat::expect_error(setTransformer(
    dimHidden = 256,
    dimHiddenRatio = 4 / 3
  ), "cannot be both set or both NULL", fixed = TRUE)
})

test_that("numerical embedding works as expected", {
  skip_if_no_python()
  embeddings <- 32L # size of embeddings
  features <- 2L # number of numerical features
  patients <- 9L

  featureIds <- torch$cat(c(
    torch$ones(9L, 1L, dtype = torch$long),
    torch$ones(9L, 1L, dtype = torch$long) * 2L
  ), dim = 1L)
  values <- torch$randn(c(patients, features))

  numericalEmbeddingClass <-
    importModule("Embeddings")$NumericalEmbedding
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
  # needs temporal covariates from FeatureExtraction
  skip_if_no_integration()
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
      analysisPath = tempdir()
    )

  expect_equal(class(results), "plpModel")
  expect_equal(attr(results, "modelType"), "binary")
  expect_equal(attr(results, "saveType"), "file")

  # check prediction between 0 and 1
  expect_gt(min(results$prediction$value), 0)
  expect_lt(max(results$prediction$value), 1)
})
