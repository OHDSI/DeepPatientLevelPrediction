catFeatures <- dataset$numCatFeatures()
numFeatures <- dataset$numNumFeatures()

modelType <- ResNet

modelParameters <- list(
  catFeatures = catFeatures,
  numFeatures = numFeatures,
  sizeEmbedding = 16,
  sizeHidden = 16,
  numLayers = 2,
  hiddenFactor = 2
)

estimatorSettings <- setEstimator(learningRate = 3e-4,
                                 weightDecay = 0.0,
                                 batchSize = 128,
                                 epochs = 1,
                                 device = 'cpu')

estimator <- Estimator$new(
  modelType = modelType,
  modelParameters = modelParameters,
  estimatorSettings = estimatorSettings
)

test_that("Estimator initialization works", {

  # count parameters in both instances
  testthat::expect_equal(
    sum(sapply(estimator$model$parameters, function(x) prod(x$shape))),
    sum(sapply(do.call(modelType, modelParameters)$parameters, function(x) prod(x$shape)))
  )

  testthat::expect_equal(
    estimator$modelParameters,
    modelParameters
  )

  # check the function that results the value from a list
  val <- estimator$itemOrDefaults(list(param = 1, test = 3), "param", default = NULL)
  expect_equal(val, 1)
  val <- estimator$itemOrDefaults(list(param = 1, test = 3), "paramater", default = NULL)
  expect_true(is.null(val))
})
sink(nullfile())
estimator$fit(dataset, dataset)
sink()

test_that("estimator fitting works", {


  # check the fitting
  # estimator$fitEpoch(dataset, batchIndex)
  # estimator$finishFit(valAUCs, modelStateDict, valLosses, epoch)
  # estimator$score(dataset, batchIndex)
  expect_true(!is.null(estimator$bestEpoch))
  expect_true(!is.null(estimator$bestScore$loss))
  expect_true(!is.null(estimator$bestScore$auc))

  old_weights <- estimator$model$head$weight$mean()$item()

  sink(nullfile())
  estimator$fitWholeTrainingSet(dataset, estimator$learnRateSchedule)
  sink()

  expect_equal(estimator$optimizer$param_groups[[1]]$lr, tail(estimator$learnRateSchedule, 1)[[1]])

  new_weights <- estimator$model$head$weight$mean()$item()

  # model should be updated when refitting
  expect_true(old_weights != new_weights)

  estimator$save(testLoc, "estimator.pt")

  expect_true(file.exists(file.path(testLoc, "estimator.pt")))

  preds <- estimator$predictProba(dataset)

  expect_lt(max(preds), 1)
  expect_gt(min(preds), 0)

  classes <- estimator$predict(dataset, threshold = 0.5)
  expect_equal(all(unique(classes) %in% c(0, 1)), TRUE)

  # not sure how to test: batchToDevice(batch)
})

test_that("early stopping works", {
  earlyStop <- EarlyStopping$new(patience = 3, delta = 0, verbose = FALSE)
  testthat::expect_true(is.null(earlyStop$bestScore))
  testthat::expect_false(earlyStop$earlyStop)
  earlyStop$call(0.5)
  testthat::expect_equal(earlyStop$bestScore, 0.5)
  earlyStop$call(0.4)
  earlyStop$call(0.4)
  earlyStop$call(0.4)
  testthat::expect_true(earlyStop$earlyStop)
})

modelSettings <- setResNet(
  numLayers = 1, sizeHidden = 16, hiddenFactor = 1,
  residualDropout = 0, hiddenDropout = 0,
  sizeEmbedding = 16, hyperParamSearch = "random",
  randomSample = 1,
  setEstimator(epochs=1,
               learningRate = 3e-4)
)

sink(nullfile())
results <- fitEstimator(trainData$Train, modelSettings = modelSettings, analysisId = 1)
sink()

test_that("Estimator fit function works", {
  expect_true(!is.null(results$trainDetails$trainingTime))

  expect_equal(class(results), "plpModel")
  expect_equal(attr(results, "modelType"), "binary")
  expect_equal(attr(results, "saveType"), "file")
})

test_that("predictDeepEstimator works", {

  # input is an estimator and a dataset
  sink(nullfile())
  predictions <- predictDeepEstimator(estimator, dataset, cohort = trainData$Train$labels)
  sink()

  expect_lt(max(predictions$value), 1)
  expect_gt(min(predictions$value), 0)
  expect_equal(nrow(predictions), nrow(trainData$Train$labels))

  # input is a plpModel and data
  sink(nullfile())
  predictions <- predictDeepEstimator(
    plpModel = results, data = trainData$Test,
    trainData$Test$labels
  )
  sink()
  expect_lt(max(predictions$value), 1)
  expect_gt(min(predictions$value), 0)
  expect_equal(nrow(predictions), nrow(trainData$Test$labels))
})

test_that("batchToDevice works", {
  # since we can't test moving to gpu there is a fake device called meta
  # which we can use to test of the device is updated
  estimator$device <- "meta"
  b <- 1:10
  batch <- estimator$batchToDevice(dataset[b])

  devices <- lapply(
    lapply(unlist(batch, recursive = TRUE), function(x) x$device),
    function(x) x == torch::torch_device(type = "meta")
  )
  # test that all are meta
  expect_true(all(devices == TRUE))
})


# cases to add, estimator with early stopping that stops, and estimator without earlystopping
