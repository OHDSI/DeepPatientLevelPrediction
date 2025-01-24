resNetSettings <- setResNet(numLayers = c(1, 2, 4),
                            sizeHidden = 64,
                            hiddenFactor = 1,
                            residualDropout = 0.5,
                            hiddenDropout = 0.5,
                            sizeEmbedding = 64,
                            estimatorSettings =
                              setEstimator(learningRate = 3e-4,
                                           weightDecay = 1e-3,
                                           device = "cpu",
                                           batchSize = 64,
                                           epochs = 1,
                                           seed = 42),
                            hyperParamSearch = "random",
                            randomSample = 3,
                            randomSampleSeed = 42)

trainCache <- trainingCache$new(testLoc)
paramSearch <- resNetSettings$param

test_that("Training cache exists on disk", {
  testthat::expect_true(
                        file.exists(file.path(testLoc, "paramPersistence.rds")))
})

test_that("Grid search can be cached", {
  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- length(paramSearch)
  trainCache$saveGridSearchPredictions(gridSearchPredictons)

  index <- 1

  gridSearchPredictons[[index]] <- list(
    prediction = list(NULL),
    param = paramSearch[[index]]
  )
  trainCache$saveGridSearchPredictions(gridSearchPredictons)

  testthat::expect_identical(trainCache$getGridSearchPredictions(),
                             gridSearchPredictons)
  testthat::expect_equal(trainCache$getLastGridSearchIndex(), index + 1)
})

test_that("Param grid predictions can be cached", {
  testthat::expect_false(trainCache$isParamGridIdentical(paramSearch))

  trainCache$saveModelParams(paramSearch)
  testthat::expect_true(trainCache$isParamGridIdentical(paramSearch))
})

test_that("Estimator can resume training from cache", {
  trainCache <- readRDS(file.path(fitEstimatorPath, "paramPersistence.rds"))
  newPath <- file.path(testLoc, "resume")
  dir.create(newPath)

  # remove last row
  trainCache$gridSearchPredictions[[2]] <- NULL
  length(trainCache$gridSearchPredictions) <- 2

  # save new cache
  saveRDS(trainCache, file = file.path(newPath, "paramPersistence.rds"))

  sink(nullfile())
  fitEstimatorResults <- fitEstimator(trainData$Train,
                                      modelSettings = modelSettings,
                                      analysisId = 1,
                                      analysisPath = newPath)
  sink()
  newCache <- readRDS(file.path(newPath, "paramPersistence.rds"))
  cS <- nrow(newCache$gridSearchPredictions[[2]]$gridPerformance$hyperSummary)
  testthat::expect_equal(cS, 4)
})

test_that("Prediction is cached for optimal parameters", {
  testCache <- readRDS(file.path(fitEstimatorPath, "paramPersistence.rds"))
  indexOfMax <-
    which.max(unlist(lapply(testCache$gridSearchPredictions,
                            function(x) x$gridPerformance$cvPerformance)))
  indexOfMin <-
    which.min(unlist(lapply(testCache$gridSearchPredictions,
                            function(x) x$gridPerformance$cvPerformance)))
  myClass <- class(testCache$gridSearchPredictions[[indexOfMax]]$prediction)
  testthat::expect_equal(myClass, class(data.frame()))
  lowestIndex <- testCache$gridSearchPredictions[[indexOfMin]]$prediction[[1]]
  testthat::expect_null(lowestIndex)
})
