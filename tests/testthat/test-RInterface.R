test_that("package loading and settings do not initialize Python", {
  # other test files may have started Python already
  skip_if(.pythonRequested)

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
      "tqdm"
    )
  )
  expect_identical(calls$pythonVersion, ">=3.11")
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
  missingPath <- tempfile("missing-deep-plp-model")

  expect_error(setFinetuner(missingPath), "does not exist")
  expect_error(setCustomEmbeddingModel(missingPath), "does not exist")
})

test_that("batched matrix multiplication does not crash (#191)", {
  skip_if_no_python()

  result <- torch$bmm(torch$randn(4L, 16L, 32L), torch$randn(4L, 32L, 16L))

  expect_equal(result$shape[0], 4L)
  expect_equal(result$shape[1], 16L)
  expect_equal(result$shape[2], 16L)
})

test_that("a message reports when torch was imported without RTLD_DEEPBIND", {
  skip_if_no_python()
  skip_if_not(reticulate::py_has_attr(reticulate::import("os"), "RTLD_DEEPBIND"))

  importedTorch <- reticulate::import("torch")
  marker <- "_dplp_deepbind"
  expect_true(reticulate::py_has_attr(importedTorch, marker))
  expect_no_message(.importTorchWithDeepBind())

  builtins <- reticulate::import_builtins()
  builtins$delattr(importedTorch, marker)
  on.exit(builtins$setattr(importedTorch, marker, TRUE), add = TRUE)
  expect_message(.importTorchWithDeepBind(), "could not be applied")
})

test_that("setEstimator turns on the output bias initialization by default", {
  settings <- setEstimator(seed = 1)
  expect_true(settings$initOutputBias)
  expect_false(setEstimator(seed = 1, initOutputBias = FALSE)$initOutputBias)
  expect_error(setEstimator(seed = 1, initOutputBias = "yes"), "initOutputBias is wrong class", fixed = TRUE)
  expect_named(camelCaseToSnakeCaseNames(list(initOutputBias = TRUE)), "init_output_bias")
})

test_that("setEstimator checks the gradient clipping and deterministic settings", {
  settings <- setEstimator(seed = 1)
  expect_null(settings$gradClipNorm)
  expect_false(settings$deterministic)
  settings <- setEstimator(seed = 1, gradClipNorm = 1, deterministic = TRUE)
  expect_equal(settings$gradClipNorm, 1)
  expect_true(settings$deterministic)
  # several values are tuned like other estimator settings
  expect_equal(
    setEstimator(seed = 1, gradClipNorm = c(1, 5))$paramsToTune,
    list(estimator.gradClipNorm = c(1, 5))
  )
  for (gradClipNorm in list(0, -1, NA_real_, Inf)) {
    expect_error(
      setEstimator(seed = 1, gradClipNorm = gradClipNorm),
      "gradClipNorm must be NULL or positive numbers"
    )
  }
  expect_error(setEstimator(seed = 1, gradClipNorm = "1"), "gradClipNorm is wrong class", fixed = TRUE)
  for (deterministic in list(NA, c(TRUE, FALSE))) {
    expect_error(
      setEstimator(seed = 1, deterministic = deterministic),
      "deterministic must be TRUE or FALSE"
    )
  }
  expect_named(
    camelCaseToSnakeCaseNames(list(gradClipNorm = 1, deterministic = TRUE)),
    c("grad_clip_norm", "deterministic")
  )
})
