skip_if_no_python()

resNet <- importModule("ResNet")$ResNet

test_that("LR scheduler that changes per batch works", {
  model <- resNet(
    feature_info = simDataset$get_feature_info(),
    size_embedding = 32L, size_hidden = 64L,
    num_layers = 1L, hidden_factor = 1L
  )
  optimizer <- torch$optim$AdamW(model$parameters(), lr = 1e-7)


  exponentialSchedulerPerBatch <-
    reticulate::import_from_path("LrFinder",
      path = pythonPath
    )$ExponentialSchedulerPerBatch
  scheduler <- exponentialSchedulerPerBatch(optimizer,
    end_lr = 1e-2,
    num_iter = 5
  )
  expect_equal(scheduler$last_epoch, 0)
  expect_equal(scheduler$optimizer$param_groups[[1]]$lr, 1e-7)

  for (i in 1:5) {
    optimizer$step()
    scheduler$step()
  }

  expect_equal(scheduler$last_epoch, 5)
  expect_equal(
    scheduler$optimizer$param_groups[[1]]$lr,
    (1e-7 * (0.01 / 1e-7)^(5 / 4))
  )
})


test_that("LR finder works", {
  estimatorSettings <- setEstimator(
    batchSize = 32L,
    seed = 42
  )
  modelParameters <- list(
    feature_info = simDataset$get_feature_info(),
    size_embedding = 32L,
    size_hidden = 64L,
    num_layers = 1L,
    hidden_factor = 1L,
    modelType = "ResNet"
  )
  estimatorSettings <- estimatorSettings
  lrSettings <- list(
    minLr = 1e-8,
    maxLr = 0.01,
    numLr = 20L,
    divergenceThreshold = 1.1
  )
  # initial LR should be the minLR
  parameters <- list(
    modelParameters = modelParameters,
    estimatorSettings = estimatorSettings
  )
  lr <- getLR(
    parameters = parameters,
    lrSettings = lrSettings,
    dataset = simDataset
  )
  tol <- 1e-10
  expect_lte(lr, 0.01 + tol)
  expect_gte(lr, 1e-08 - tol)

  # following the LR finder, the actual training should use the LR found
  modelSettings <- setResNet(
    numLayers = 1, sizeHidden = 16, hiddenFactor = 1,
    residualDropout = 0, hiddenDropout = 0,
    sizeEmbedding = 16, hyperParamSearch = "random",
    randomSample = 1,
    setEstimator(
      epochs = 1,
      learningRate = "auto"
    )
  )
  results <- fitEstimator(
    simTrainData$Train,
    modelSettings, 1,
    withr::local_tempdir()
  )
  expect_false(results$trainDetails$finalModelParameters$learnSchedule$LRs == 3e-4)
})

test_that("LR finder works with device specified by a function", {
  deviceFun <- function() {
    dev <- "cpu"
    dev
  }
  modelParameters <-
    list(
      feature_info = simDataset$get_feature_info(),
      size_embedding = 8L,
      size_hidden = 16L,
      num_layers = 1L,
      hidden_factor = 1L,
      modelType = "ResNet"
    )
  estimatorSettings <- setEstimator(
    batchSize = 32L,
    seed = 42,
    device = deviceFun
  )
  lrSettings <- list(
    minLr = 1e-6,
    maxLr = 0.03,
    numLr = 20L,
    divergenceThreshold = 1.1
  )
  parameters <- list(
    modelParameters = modelParameters,
    estimatorSettings = estimatorSettings
  )
  lr <- getLR(
    parameters = parameters,
    lrSettings = lrSettings,
    dataset = simDataset
  )

  tol <- 1e-8
  expect_lte(lr, 0.03 + tol)
  expect_gte(lr, 1e-6 - tol)
})

test_that("an automatic learning rate only sees the training folds", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  settings <- createMockSettings(learningRate = "auto")
  dataset <- createDataset(data, labels)
  parameters <- settings$param[[1]]
  searched <- list()
  testthat::with_mocked_bindings(
    doCrossValidationImpl(dataset, labels, parameters, settings),
    getLR = function(parameters, dataset, lrSettings = NULL) {
      searched[[length(searched) + 1]] <<- sort(unlist(dataset$indices))
      0.01
    }
  )
  expect_equal(searched, list(c(2L, 3L, 6L, 7L), c(0L, 1L, 4L, 5L)))
})

test_that("the learning rate finder clears gradients between trials", {
  skip_if_no_python()
  toy <- toyPython()
  estimator <- toy$toy_estimator(
    importModule("Estimator")$Estimator, optimizer = torch$optim$AdamW,
    weight_decay = 0
  )
  finder <- importModule("LrFinder")$LrFinder(
    estimator, lr_settings = list(num_lr = 2L, min_lr = 0.01, max_lr = 0.1)
  )
  dataset <- toy$ToyData(c(1, 1), c(1, 1))
  finder$get_lr(dataset)
  # after the last trial the gradient has been cleared
  expect_null(estimator$model$weight$grad)
})

test_that("the learning rate finder handles an early loss minimum", {
  skip_if_no_python()
  suggest <- importModule("LrFinder")$suggest_lr_index
  # The smallest loss is at python index `minimum`. Before it, the steepest decrease is
  # from the first to the second step. After it, a steeper drop that stays above the
  # minimum would be picked by a search that runs past the minimum.
  lossesWithMinimumAt <- function(minimum, n = 30) {
    descent <- if (minimum == 1) 1 else seq(4, 1, length.out = minimum)
    after <- c(seq(2, 20, length.out = 5), 1.5, seq(3, 9, length.out = n - minimum - 7))
    torch$tensor(c(10, descent, after))
  }
  # minima at the 4th and 5th step left no losses to search and argmin failed
  expect_equal(suggest(lossesWithMinimumAt(4L), 5L), 0L)
  expect_equal(suggest(lossesWithMinimumAt(5L), 5L), 0L)
  # minima at the 1st to 3rd step wrapped around to the end of the search
  for (minimum in 1:3) {
    expect_lt(suggest(lossesWithMinimumAt(minimum), 5L), minimum)
  }
  # a later minimum keeps the margin of min_factor steps
  losses <- torch$tensor(c(10, 9, 8, 7, 6, 2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 3, 5))
  expect_equal(suggest(losses, 5L), 4L)
  expect_equal(suggest(torch$tensor(c(1, 2, 3)), 5L), 0L)
})
