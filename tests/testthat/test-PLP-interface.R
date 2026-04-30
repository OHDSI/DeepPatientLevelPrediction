test_that("deep model settings expose the PLP model interface", {
  settings <- setRealMLP(numLayers = c(2L, 3L))

  expect_equal(
    settings$fitFunction,
    "DeepPatientLevelPrediction::fitDeepPlpClassifier"
  )
  expect_named(
    settings$settings,
    c(
      "modelName", "modelType", "deepModelType", "estimatorSettings",
      "prepareData", "train", "predict", "variableImportance", "saveType",
      "requiresDenseMatrix", "seed"
    )
  )
  expect_equal(settings$settings$modelType, "binary")
  expect_equal(settings$settings$deepModelType, "RealMLP")
  expect_equal(length(settings$paramDefinition$numLayers), 2L)
})

test_that("legacy helper search arguments warn when used explicitly", {
  withr::local_options(list(
    DeepPatientLevelPrediction.suppressLegacySearchWarning = FALSE
  ))

  expect_warning(
    setRealMLP(hyperParamSearch = "random", randomSample = 1L),
    "deprecated"
  )
})

test_that("new PLP hyperparameter settings do not warn for default helper args", {
  settings <- setRealMLP()
  hyperparameterSettings <- PatientLevelPrediction::createHyperparameterSettings(
    search = "random",
    sampleSize = 1L,
    randomSeed = 42L
  )

  expect_warning(
    resolveHyperparameterSettings(settings, hyperparameterSettings),
    NA
  )
})

test_that("default legacy random sampling does not reject small grids", {
  settings <- setResNet(
    numLayers = 1L,
    sizeHidden = 16L,
    hiddenFactor = 1L,
    residualDropout = 0,
    hiddenDropout = 0,
    sizeEmbedding = 16L
  )

  expect_length(settings$param, 2L)
})

test_that("PLP random settings operate on raw parameter definitions", {
  settings <- setRealMLP(
    numLayers = c(2L, 3L),
    sizeHidden = c(64L, 128L)
  )
  hyperparameterSettings <- PatientLevelPrediction::createHyperparameterSettings(
    search = "random",
    sampleSize = 2L,
    randomSeed = 42L
  )

  candidates <- createCandidatePool(
    paramDefinition = settings$paramDefinition,
    hyperparameterSettings = hyperparameterSettings
  )

  expect_length(candidates, 2L)
  expect_true(all(vapply(candidates, function(x) {
    all(c("numLayers", "sizeHidden") %in% names(x))
  }, logical(1))))
})

test_that("custom tuning metrics are used for CV performance", {
  prediction <- data.frame(
    rowId = 1:4,
    outcomeCount = c(0, 1, 0, 1),
    index = c(1, 1, 2, 2),
    value = c(0.1, 0.2, 0.7, 0.8)
  )
  metric <- PatientLevelPrediction::createTuningMetric(
    fun = function(prediction) mean(prediction$value),
    maximize = FALSE,
    name = "meanPrediction"
  )

  performance <- computeDeepGridPerformance(
    prediction = prediction,
    parameters = list(numLayers = 1L),
    tuningMetric = metric
  )

  expect_equal(performance$metric, "meanPrediction")
  expect_equal(performance$cvPerformancePerFold, c(0.15, 0.75))
  expect_equal(performance$cvPerformance, 0.45)
})

test_that("adaptive custom generators must expose cache state", {
  hyperparameterSettings <- PatientLevelPrediction::createHyperparameterSettings(
    search = "custom",
    generator = function(...) list()
  )
  hyperparameterSettings$generator <- list(
    initialize = function(definition, settings) invisible(NULL),
    getNext = function(history) NULL
  )

  expect_true(isAdaptiveHyperparameterSettings(hyperparameterSettings))
  expect_error(
    gridCvDeepAdaptive(
      mappedData = NULL,
      labels = NULL,
      modelSettings = list(),
      hyperparameterSettings = hyperparameterSettings,
      modelLocation = tempdir(),
      analysisPath = tempdir(),
      paramDefinition = list(numLayers = 1L)
    ),
    "saveState"
  )
})

test_that("static PLP search cache detects truncated results", {
  cachePath <- tempfile("dplp-cache-")
  dir.create(cachePath)
  cache <- trainingCache$new(cachePath)
  cache$saveCandidatePool(list(list(a = 1L), list(a = 2L)))
  cache$saveSearchResults(list(list(
    gridPerformance = list(cvPerformance = 0.1)
  )))

  expect_false(cache$isSearchFull())
  expect_equal(cache$getNextSearchIndex(), 2L)
})

test_that("saved covariate references keep join key types stable", {
  covariateRef <- data.frame(
    analysisId = c(1, 2),
    columnId = c(1, 2),
    covariateId = c(1001, 1002)
  )
  covariateRef$analysisId <- as.numeric(covariateRef$analysisId)
  covariateRef$columnId <- as.numeric(covariateRef$columnId)

  normalized <- normalizeCovariateReferenceTypes(covariateRef)

  expect_type(normalized$analysisId, "integer")
  expect_type(normalized$columnId, "integer")
})
