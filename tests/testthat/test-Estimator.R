catFeatures <- small_dataset$dataset$numCatFeatures()
numFeatures <- small_dataset$dataset$numNumFeatures()

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
                                 epochs = 5,
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
estimator$fit(small_dataset, small_dataset)
sink()

test_that("estimator fitting works", {

  expect_true(!is.null(estimator$bestEpoch))
  expect_true(!is.null(estimator$bestScore$loss))
  expect_true(!is.null(estimator$bestScore$auc))

  old_weights <- estimator$model$head$weight$mean()$item()

  sink(nullfile())
  estimator$fitWholeTrainingSet(small_dataset, estimator$learnRateSchedule)
  sink()

  expect_equal(estimator$optimizer$param_groups[[1]]$lr, tail(estimator$learnRateSchedule, 1)[[1]])

  new_weights <- estimator$model$head$weight$mean()$item()

  # model should be updated when refitting
  expect_true(old_weights != new_weights)

  estimator$save(testLoc, "estimator.pt")

  expect_true(file.exists(file.path(testLoc, "estimator.pt")))
  
  sink(nullfile())
  preds <- estimator$predictProba(dataset)
  sink()
  
  expect_lt(max(preds), 1)
  expect_gt(min(preds), 0)
  
  sink(nullfile())
  classes <- estimator$predict(small_dataset, threshold = 0.5)
  sink()
  expect_equal(all(unique(classes) %in% c(0, 1)), TRUE)
  
  sink(nullfile())
  classes <- estimator$predict(small_dataset$dataset)
  sink()
  expect_equal(all(unique(classes) %in% c(0, 1)), TRUE)
  
  estimatorSettings <- setEstimator(learningRate = 3e-4,
                                    weightDecay = 0.0,
                                    batchSize = 128,
                                    epochs = 5,
                                    device = 'cpu',
                                    metric= "loss")
  
  estimator <- Estimator$new(
    modelType = modelType,
    modelParameters = modelParameters,
    estimatorSettings = estimatorSettings
  )
  sink(nullfile())
  estimator$fit(small_dataset, small_dataset)
  sink()
  
  expect_equal(estimator$metric$mode, "min")
  expect_equal(estimator$metric$name, "loss")
  
  sink(nullfile())
  estimator$fitWholeTrainingSet(small_dataset, estimator$learnRateSchedule)
  sink()
  
  expect_equal(estimator$learnRateSchedule[[estimator$bestEpoch]],
               estimator$optimizer$param_groups[[1]]$lr)
  
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
results <- fitEstimator(trainData$Train, modelSettings = modelSettings, analysisId = 1, analysisPath = testLoc)
sink()

test_that("Estimator fit function works", {
  expect_true(!is.null(results$trainDetails$trainingTime))

  expect_equal(class(results), "plpModel")
  expect_equal(attr(results, "modelType"), "binary")
  expect_equal(attr(results, "saveType"), "file")
  fakeTrainData <- trainData
  fakeTrainData$train$covariateData <- list(fakeCovData <- c("Fake"))
  expect_error(fitEstimator(fakeTrainData$train, modelSettings, analysisId = 1, analysisPath = testLoc))
})

test_that("predictDeepEstimator works", {

  # input is an estimator and a dataset
  sink(nullfile())
  predictions <- predictDeepEstimator(estimator, small_dataset, cohort = trainData$Train$labels)
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
  batch <- batchToDevice(dataset[b], device=estimator$device)

  devices <- lapply(
    lapply(unlist(batch, recursive = TRUE), function(x) x$device),
    function(x) x == torch::torch_device(type = "meta")
  )
  # test that all are meta
  expect_true(all(devices == TRUE))
  
  numDevice <- batchToDevice(dataset[b]$batch$num, device=estimator$device)
  expect_true(numDevice$device==torch::torch_device(type="meta"))
})

test_that("Estimator without earlyStopping works", {
  # estimator without earlyStopping
  estimatorSettings <- setEstimator(learningRate = 3e-4,
                                    weightDecay = 0.0,
                                    batchSize = 128,
                                    epochs = 1,
                                    device = 'cpu',
                                    earlyStopping = NULL)
  estimator2 <- Estimator$new(
    modelType = modelType,
    modelParameters = modelParameters,
    estimatorSettings = estimatorSettings
  )
  sink(nullfile())
  estimator2$fit(small_dataset, small_dataset)
  sink()
  
  expect_null(estimator2$earlyStopper)
  expect_true(!is.null(estimator2$bestEpoch))
  
})

test_that("Early stopper can use loss and stops early", {
  estimatorSettings <- setEstimator(learningRate = 3e-2,
                                    weightDecay = 0.0,
                                    batchSize = 128,
                                    epochs = 10,
                                    device = 'cpu',
                                    earlyStopping =list(useEarlyStopping=TRUE,
                                                        params = list(mode=c('min'), 
                                                                      patience=1)),
                                    metric = 'loss',
                                    seed=42)
  estimator <- Estimator$new(
    modelType=modelType,
    modelParameters = modelParameters,
    estimatorSettings = estimatorSettings
  )
  
  sink(nullfile())
  estimator$fit(small_dataset, small_dataset)
  sink()
  
  expect_true(estimator$bestEpoch < estimator$epochs)
  
})

test_that('Custom metric in estimator works', {
  
  metric_fun <- function(predictions, labels)  {
    positive <- predictions[labels == 1]
    negative <- predictions[labels == 0]
    pr <- PRROC::pr.curve(scores.class0 = positive, scores.class1 = negative)
    auprc <- pr$auc.integral
  }
    
  estimatorSettings <- setEstimator(learningRate = 3e-4,
                                    weightDecay = 0.0,
                                    batchSize = 128,
                                    device = "cpu",
                                    epochs = 1,
                                    metric=list(fun=metric_fun,
                                                name="auprc",
                                                mode="max"))
  
  estimator <- Estimator$new(modelType = modelType,
                             modelParameters = modelParameters,
                             estimatorSettings = estimatorSettings)
  expect_true(is.function(estimator$metric$fun))
  expect_true(is.character(estimator$metric$name))
  expect_true(estimator$metric$mode %in% c("min", "max"))
  
  sink(nullfile())
  estimator$fit(small_dataset, small_dataset)
  sink()
  
  expect_true(estimator$bestScore[["auprc"]]>0)

})

test_that("setEstimator with paramsToTune is correctly added to hyperparameters", {
  estimatorSettings <- setEstimator(learningRate=c(3e-4,1e-3),
                                    batchSize=128,
                                    epochs=1,
                                    device="cpu",
                                    metric=c("auc", "auprc"),
                                    earlyStopping = list(useEarlyStopping=TRUE,
                                                         params=list(patience=c(4,10))))
  modelSettings <- setResNet(numLayers = 1, sizeHidden = 64,
                             hiddenFactor = 1, residualDropout = 0.2,
                             hiddenDropout = 0.2,sizeEmbedding = 128,
                             estimatorSettings = estimatorSettings,
                             hyperParamSearch = "grid")
  
  expect_true(length(modelSettings$param) == 8)
  expect_true(length(unique(lapply(modelSettings$param, function(x) x$estimator.metric)))==2)
  expect_true(length(unique(lapply(modelSettings$param, function(x) x$estimator.learningRate)))==2)
  expect_true(length(unique(lapply(modelSettings$param, function(x) x$estimator.earlyStopping.patience)))==2)
  fitParams <- names(modelSettings$param[[1]])[grepl("^estimator", names(modelSettings$param[[1]]))]
  
  estimatorSettings2 <- fillEstimatorSettings(estimatorSettings, fitParams, paramSearch=modelSettings$param[[8]])
  
  expect_equal(estimatorSettings2$learningRate, 1e-3)
  expect_equal(as.character(estimatorSettings2$metric), "auprc")
  expect_equal(estimatorSettings2$earlyStopping$params$patience, 10)
})