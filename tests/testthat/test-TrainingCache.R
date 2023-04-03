resNetSettings <- setResNet(numLayers = c(1, 2, 4),
                            sizeHidden = 2^6,
                            hiddenFactor = 1,
                            residualDropout = 0.5,
                            hiddenDropout = 0.5,
                            sizeEmbedding = 2^6,
                            estimatorSettings = setEstimator(learningRate='auto',
                                                             weightDecay=1e-3,
                                                             device='cpu',
                                                             batchSize=1024,
                                                             epochs=30,
                                                             seed=NULL),
                            hyperParamSearch = "random",
                            randomSample = 2,
                            randomSampleSeed = NULL)

trainCache <- TrainingCache$new(testLoc)
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

  testthat::expect_identical(trainCache$getGridSearchPredictions(), gridSearchPredictons)
  testthat::expect_equal(trainCache$getLastGridSearchIndex(), index+1)
})

test_that("Param grid predictions can be cached", {
  testthat::expect_false(trainCache$isParamGridIdentical(paramSearch))
  
  trainCache$saveModelParams(paramSearch)
  testthat::expect_true(trainCache$isParamGridIdentical(paramSearch))
})