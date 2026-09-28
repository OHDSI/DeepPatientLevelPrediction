test_that("package loading and settings do not initialize Python", {
  skip_if(.integrationRequested)

  expect_false(reticulate::py_available(initialize = FALSE))

  estimator <- setEstimator(seed = 1)
  expect_s3_class(estimator$optimizer, "delayed")
  expect_s3_class(estimator$scheduler, "delayed")
  expect_s3_class(estimator$criterion, "delayed")
  expect_false(reticulate::py_available(initialize = FALSE))
})

test_that("Python bindings declare requirements and delay the torch import", {
  calls <- new.env(parent = emptyenv())
  torchModule <- new.env(parent = emptyenv())
  initializePythonBindings <- getFromNamespace(
    ".initializePythonBindings",
    "DeepPatientLevelPrediction"
  )
  pythonImportError <- getFromNamespace(
    ".pythonImportError",
    "DeepPatientLevelPrediction"
  )

  result <- initializePythonBindings(
    pyRequire = function(packages, python_version) {
      calls$packages <- packages
      calls$pythonVersion <- python_version
    },
    pyImport = function(module, delay_load) {
      calls$module <- module
      calls$delayLoad <- delay_load
      torchModule
    }
  )

  expect_identical(result, torchModule)
  expect_setequal(
    calls$packages,
    c(
      "polars>=1.31.0",
      "pyarrow",
      "duckdb",
      "numpy",
      "torch>=2.7,<3",
      "tqdm",
      "nvidia-ml-py"
    )
  )
  expect_identical(calls$pythonVersion, ">=3.10")
  expect_identical(calls$module, "torch")
  expect_identical(calls$delayLoad$on_error, pythonImportError)
})

test_that("Python import failures include installation guidance", {
  pythonImportError <- getFromNamespace(
    ".pythonImportError",
    "DeepPatientLevelPrediction"
  )

  error <- expect_error(
    pythonImportError(simpleError("No module named torch")),
    "Original error: No module named torch"
  )
  expect_null(error$call)
})

test_that("estimator settings are created and validated in R", {
  estimator <- setEstimator(
    learningRate = 0.001,
    weightDecay = 0.01,
    batchSize = 32,
    epochs = 2,
    seed = 42
  )

  expect_equal(estimator$learningRate, 0.001)
  expect_equal(estimator$batchSize, 32)
  expect_equal(estimator$seed, 42)
  expect_error(setEstimator(batchSize = 0), "needs to be")
  expect_error(
    setEstimator(batchSize = 10, accumulationSteps = 3),
    "divisible"
  )
})

test_that("model constructors create PLP model settings in R", {
  estimator <- setEstimator(learningRate = 0.001, seed = 42)

  mlp <- setMultiLayerPerceptron(
    numLayers = 1,
    sizeHidden = 16,
    dropout = 0.1,
    sizeEmbedding = 8,
    estimatorSettings = estimator,
    randomSample = 1
  )
  resnet <- setResNet(
    numLayers = 1,
    sizeHidden = 16,
    hiddenFactor = 1,
    residualDropout = 0.1,
    hiddenDropout = 0.1,
    sizeEmbedding = 8,
    estimatorSettings = estimator,
    randomSample = 1
  )
  transformer <- setTransformer(
    numBlocks = 1,
    dimToken = 8,
    numHeads = 2,
    dimHidden = 16,
    estimatorSettings = estimator,
    randomSample = 1
  )
  realMlp <- setRealMLP(
    numLayers = 1,
    sizeHidden = 16,
    sizeEmbedding = 8
  )

  expect_s3_class(mlp, "modelSettings")
  expect_s3_class(resnet, "modelSettings")
  expect_s3_class(transformer, "modelSettings")
  expect_s3_class(realMlp, "modelSettings")
  expect_equal(mlp$modelType, "MultiLayerPerceptron")
  expect_equal(resnet$modelType, "ResNet")
  expect_equal(transformer$modelType, "Transformer")
  expect_equal(realMlp$modelType, "RealMLP")
})

test_that("default and temporal model settings are available without Python", {
  defaultResnet <- setDefaultResNet()
  defaultTransformer <- setDefaultTransformer()
  temporalTransformer <- setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      positionalEncoding = "SinusoidalPE",
      maxSequenceLength = 128,
      truncation = "tail",
      timeTokens = TRUE
    )
  )

  expect_equal(defaultResnet$param[[1]]$numLayers, 6)
  expect_equal(defaultTransformer$param[[1]]$numBlocks, 3)
  expect_true(isTRUE(attr(temporalTransformer$param, "temporalModel")))
  expect_equal(
    attr(temporalTransformer$param, "temporalSettings")$maxSequenceLength,
    128L
  )
})

test_that("training cache persists R objects", {
  cacheDirectory <- withr::local_tempdir()
  cache <- trainingCache$new(cacheDirectory)
  parameters <- list(list(sizeHidden = 16), list(sizeHidden = 32))
  predictions <- list(
    list(gridPerformance = list(cvPerformance = 0.6), prediction = 1),
    list(gridPerformance = list(cvPerformance = 0.7), prediction = 2)
  )

  expect_true(file.exists(file.path(cacheDirectory, "paramPersistence.rds")))
  expect_false(cache$isParamGridIdentical(parameters))
  cache$saveModelParams(parameters)
  cache$saveGridSearchPredictions(predictions)
  expect_true(cache$isParamGridIdentical(parameters))
  expect_identical(cache$getGridSearchPredictions(), predictions)
  expect_true(cache$isFull())

  restored <- trainingCache$new(cacheDirectory)
  expect_identical(restored$getGridSearchPredictions(), predictions)
})

test_that("Python-backed settings reject missing model files before import", {
  missingPath <- file.path(tempdir(), "missing-deep-plp-model")

  expect_error(setFinetuner(missingPath), "does not exist")
  expect_error(setCustomEmbeddingModel(missingPath), "does not exist")
})
