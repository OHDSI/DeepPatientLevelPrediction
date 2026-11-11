resNetSettings <- setResNet(
  numLayers = c(1, 2, 4),
  sizeHidden = 64,
  hiddenFactor = 1,
  residualDropout = 0.5,
  hiddenDropout = 0.5,
  sizeEmbedding = 64,
  estimatorSettings =
    setEstimator(
      learningRate = 3e-4,
      weightDecay = 1e-3,
      device = "cpu",
      batchSize = 64,
      epochs = 1,
      seed = 42
    ),
  hyperParamSearch = "random",
  randomSample = 3,
  randomSampleSeed = 42
)

paramSearch <- resNetSettings$param

test_that("Training cache exists on disk", {
  cacheDirectory <- withr::local_tempdir()
  trainingCache$new(cacheDirectory)
  expect_true(
    file.exists(file.path(cacheDirectory, "paramPersistence.rds"))
  )
})

test_that("Grid search can be cached", {
  trainCache <- trainingCache$new(withr::local_tempdir())
  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- length(paramSearch)
  trainCache$saveGridSearchPredictions(gridSearchPredictons)

  index <- 1

  gridSearchPredictons[[index]] <- list(
    prediction = list(NULL),
    param = paramSearch[[index]]
  )
  trainCache$saveGridSearchPredictions(gridSearchPredictons)

  expect_identical(
    trainCache$getGridSearchPredictions(),
    gridSearchPredictons
  )
  expect_equal(trainCache$getLastGridSearchIndex(), index + 1)
})

test_that("Param grid predictions can be cached", {
  trainCache <- trainingCache$new(withr::local_tempdir())
  expect_false(trainCache$isParamGridIdentical(paramSearch))

  trainCache$saveModelParams(paramSearch)
  expect_true(trainCache$isParamGridIdentical(paramSearch))
})

test_that("Estimator can resume training from cache", {
  skip_if_no_python()
  # fitting the model writes the cache
  force(simFitEstimatorResults)
  trainCache <- readRDS(file.path(simFitEstimatorPath, "paramPersistence.rds"))
  newPath <- withr::local_tempdir()
  dir.create(newPath, showWarnings = FALSE)

  # remove last row
  trainCache$gridSearchPredictions[[2]] <- NULL
  length(trainCache$gridSearchPredictions) <- 2

  # save new cache
  saveRDS(trainCache, file = file.path(newPath, "paramPersistence.rds"))

  quietly({
    fitEstimatorResults <- fitEstimator(simTrainData$Train,
      modelSettings = simModelSettings,
      analysisId = 1,
      analysisPath = newPath
    )
  })
  newCache <- readRDS(file.path(newPath, "paramPersistence.rds"))
  cS <- nrow(newCache$gridSearchPredictions[[2]]$gridPerformance$hyperSummary)
  expect_equal(cS, 4)
})

test_that("Prediction is cached for optimal parameters", {
  skip_if_no_python()
  # fitting the model writes the cache
  force(simFitEstimatorResults)
  testCache <- readRDS(file.path(simFitEstimatorPath, "paramPersistence.rds"))
  indexOfMax <-
    which.max(unlist(lapply(
      testCache$gridSearchPredictions,
      function(x) x$gridPerformance$cvPerformance
    )))
  indexOfMin <-
    which.min(unlist(lapply(
      testCache$gridSearchPredictions,
      function(x) x$gridPerformance$cvPerformance
    )))
  myClass <- class(testCache$gridSearchPredictions[[indexOfMax]]$prediction)
  expect_equal(myClass, class(data.frame()))
  lowestIndex <- testCache$gridSearchPredictions[[indexOfMin]]$prediction[[1]]
  expect_null(lowestIndex)
})

test_that("training cache is ignored when the data or settings change", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  settings <- createMockSettings()
  dataset <- createDataset(data, labels)
  fingerprint <- createCacheFingerprint(dataset, labels, settings)

  expect_identical(
    createCacheFingerprint(createDataset(data, labels), labels, settings),
    fingerprint
  )

  flipped <- labels
  flipped$outcomeCount <- 1 - flipped$outcomeCount
  expect_false(identical(
    createCacheFingerprint(createDataset(data, flipped), flipped, settings),
    fingerprint
  ))

  newFolds <- labels
  newFolds$index <- rev(newFolds$index)
  expect_false(identical(
    createCacheFingerprint(dataset, newFolds, settings), fingerprint
  ))

  changedSettings <- list(
    createMockSettings(learningRate = 0.1),
    createMockSettings(seed = 43),
    createMockSettings(optimizer = torch$optim$SGD)
  )
  for (changed in changedSettings) {
    expect_false(identical(
      createCacheFingerprint(dataset, labels, changed), fingerprint
    ))
  }

  otherValues <- data$covariates %>% dplyr::collect()
  otherValues$covariateValue <- otherValues$covariateValue + 1
  otherData <- createMockData(otherValues)
  withr::defer(Andromeda::close(otherData))
  expect_false(identical(
    createCacheFingerprint(createDataset(otherData, labels), labels, settings),
    fingerprint
  ))
})

test_that("a generated seed does not invalidate the training cache", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  dataset <- createDataset(data, labels)
  first <- withr::with_seed(1, createMockSettings(seed = NULL))
  second <- withr::with_seed(2, createMockSettings(seed = NULL))
  expect_false(identical(first$estimatorSettings$seed, second$estimatorSettings$seed))
  expect_identical(
    createCacheFingerprint(dataset, labels, first),
    createCacheFingerprint(dataset, labels, second)
  )
})

test_that("the fingerprint tells tuning metrics apart by function and arguments", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  settings <- createMockSettings()
  dataset <- createDataset(data, labels)
  metric <- function(fun, threshold) {
    PatientLevelPrediction::createTuningMetric(
      fun = fun, maximize = TRUE, name = "share", funArgs = list(threshold = threshold)
    )
  }
  above <- function(prediction, threshold) mean(prediction$value > threshold)
  below <- function(prediction, threshold) mean(prediction$value < threshold)
  fingerprint <- function(tuningMetric) {
    createCacheFingerprint(dataset, labels, settings, tuningMetric)
  }
  reference <- fingerprint(metric(above, 0.2))
  expect_identical(fingerprint(metric(above, 0.2)), reference)
  expect_false(identical(fingerprint(metric(above, 0.8)), reference))
  expect_false(identical(fingerprint(metric(below, 0.2)), reference))
})

test_that("the fingerprint does not depend on how equal labels are stored", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  settings <- createMockSettings()
  dataset <- createDataset(data, labels)
  # seq_len() gives a compact sequence; merge() gives an ordinary vector
  merged <- merge(labels, data.frame(rowId = labels$rowId), by = "rowId")
  expect_identical(
    createCacheFingerprint(dataset, merged, settings),
    createCacheFingerprint(dataset, labels, settings)
  )
})

test_that("the data fingerprint does not depend on the chunk size", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  dataset <- createDataset(data, createMockLabels())
  expect_identical(dataset$fingerprint(chunk_rows = 1L), dataset$fingerprint())
  expect_identical(dataset$fingerprint(chunk_rows = 3L), dataset$fingerprint())
})

test_that("files read during training are identified by their contents", {
  skip_if_no_python()
  modelFile <- withr::local_tempfile(lines = "weights 1")
  embeddingFile <- withr::local_tempfile(lines = "embeddings 1")
  strategy <- importModule("InitStrategy")$CustomEmbeddingInitStrategy(
    embedding_class = "CustomEmbeddings", embedding_file = embeddingFile
  )
  settings <- list(finetuneModelPath = modelFile, initStrategy = strategy)
  first <- describeInputFiles(settings)
  expect_length(first, 2)

  writeLines("weights 2", modelFile)
  second <- describeInputFiles(settings)
  expect_false(identical(second[1], first[1]))
  expect_identical(second[2], first[2])

  writeLines("embeddings 2", embeddingFile)
  expect_false(identical(describeInputFiles(settings)[2], first[2]))
  expect_length(describeInputFiles(list()), 0)
})

test_that("an interrupted search without a seed resumes where it stopped", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  analysisPath <- withr::local_tempdir()
  # a new seed is drawn every time the settings are created
  newSettings <- function() {
    setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = c(4, 8), dropout = 0, sizeEmbedding = 4,
      estimatorSettings = setEstimator(
        learningRate = 0.01, batchSize = 4, epochs = 1, device = "cpu", seed = NULL
      ),
      hyperParamSearch = "grid"
    )
  }
  fitted <- c()
  interrupt <- TRUE
  crossValidate <- function(dataset, labels, parameters, modelSettings, tuningMetric) {
    if (interrupt && parameters$sizeHidden == 8) {
      stop("interrupted")
    }
    fitted <<- c(fitted, parameters$sizeHidden)
    prediction <- labels
    prediction$value <- 0.5
    list(
      prediction = prediction,
      param = c(parameters, list(learnSchedule = list(LRs = list(0.01)))),
      gridPerformance = list(
        cvPerformance = parameters$sizeHidden / 10,
        hyperSummary = data.frame(metric = "x")
      )
    )
  }
  run <- function(settings) {
    testthat::with_mocked_bindings(
      gridCvDeep(
        mappedData = data, labels = labels, modelSettings = settings,
        modelLocation = withr::local_tempdir(.local_envir = parent.frame(3)),
        analysisPath = analysisPath
      ),
      doCrossValidation = crossValidate
    )
  }

  first <- withr::with_seed(1, newSettings())
  expect_error(run(first), "interrupted")
  expect_equal(fitted, 4)

  second <- withr::with_seed(2, newSettings())
  expect_false(identical(first$estimatorSettings$seed, second$estimatorSettings$seed))
  interrupt <- FALSE
  result <- run(second)
  # only the combination that had not finished is computed again
  expect_equal(fitted, c(4, 8))
  expect_equal(result$finalParam$sizeHidden, 8)
})

test_that("gridCvDeep retrains when the outcomes change and resumes when they do not", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  settings <- createMockSettings()
  analysisPath <- withr::local_tempdir()
  run <- function(labels) {
    gridCvDeep(
      mappedData = data, labels = labels, modelSettings = settings,
      modelLocation = withr::local_tempdir(.local_envir = parent.frame(2)),
      analysisPath = analysisPath
    )
  }
  cvOutcomes <- function(result) {
    cv <- result$prediction[result$prediction$evaluationType == "CV", ]
    cv$outcomeCount[order(cv$rowId)]
  }

  first <- run(labels)
  expect_equal(cvOutcomes(first), labels$outcomeCount)

  resumed <- testthat::with_mocked_bindings(
    run(labels),
    doCrossValidation = function(...) stop("the cache should have been reused")
  )
  expect_equal(cvOutcomes(resumed), labels$outcomeCount)

  flipped <- labels
  flipped$outcomeCount <- 1 - flipped$outcomeCount
  expect_equal(cvOutcomes(run(flipped)), flipped$outcomeCount)
})
