skip_if_no_python()

featureInfo <- simSmallDataset$dataset$get_feature_info()
modelParameters <- list(
  feature_info = featureInfo,
  sizeEmbedding = 16,
  sizeHidden = 16,
  numLayers = 2,
  hiddenFactor = 2,
  modelType = "ResNet"
)

estimatorSettings <-
  setEstimator(learningRate = 3e-4,
               weightDecay = 0.0,
               batchSize = 128,
               epochs = 5,
               device = "cpu",
               seed = 42,
               optimizer = torch$optim$AdamW,
               criterion = torch$nn$BCEWithLogitsLoss,
               metric = "loss",
               scheduler =
               list(fun = torch$optim$lr_scheduler$ReduceLROnPlateau,
                    params = list(patience = 1)),
               earlyStopping = NULL)
parameters <- list(modelParameters = modelParameters,
                   estimatorSettings = estimatorSettings)
# Each call builds a new estimator, so that tests cannot change each other's.
newEstimator <- function() createEstimator(parameters = parameters)
fittedEstimator <- function() {
  estimator <- newEstimator()
  quietly(estimator$fit(simSmallDataset, simSmallDataset))
  estimator
}

test_that("Estimator initialization works", {
  estimator <- newEstimator()

  # count parameters in both instances
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  resNet <-
    reticulate::import_from_path(modelParameters$modelType, 
                                 path = path)[[modelParameters$modelType]]

   testthat::expect_equal(
    sum(reticulate::iterate(estimator$model$parameters(),
                            function(x) x$numel())),
    sum(reticulate::iterate(do.call(resNet, camelCaseToSnakeCaseNames(modelParameters))$parameters(),
                            function(x) x$numel()))
  )

  testthat::expect_equal(
    estimator$model_parameters,
    camelCaseToSnakeCaseNames(modelParameters)
  )
})

test_that("Estimator detects wrong inputs", {

  testthat::expect_error(setEstimator(learningRate = "notAuto"), "should be either a numeric or \"auto\"", fixed = TRUE)
  testthat::expect_error(setEstimator(weightDecay = -1), "weightDecay needs to be >= 0", fixed = TRUE)
  testthat::expect_error(setEstimator(weightDecay = "text"), "weightDecay is wrong class", fixed = TRUE)
  testthat::expect_error(setEstimator(batchSize = 0), "batchSize needs to be > 0", fixed = TRUE)
  testthat::expect_error(setEstimator(batchSize = "text"), "batchSize is wrong class", fixed = TRUE)
  testthat::expect_error(setEstimator(epochs = 0), "epochs needs to be > 0", fixed = TRUE)
  testthat::expect_error(setEstimator(epochs = "test"), "epochs is wrong class", fixed = TRUE)
  testthat::expect_error(setEstimator(earlyStopping = "notListorNull"), "earlyStopping is wrong class", fixed = TRUE)
  testthat::expect_error(setEstimator(metric = 1), "metric is wrong class", fixed = TRUE)
  testthat::expect_error(setEstimator(seed = "32"), "seed is wrong class", fixed = TRUE)
})

test_that("estimator fitting works", {
  estimator <- fittedEstimator()

  expect_true(!is.null(estimator$best_epoch))
  expect_true(!is.null(estimator$best_score$loss))
  expect_true(!is.null(estimator$best_score$auc))

  oldWeights <- estimator$model$head$weight$mean()$item()

  quietly({
    estimator$fit_whole_training_set(simSmallDataset, estimator$learn_rate_schedule)
  })

  expect_equal(estimator$optimizer$param_groups[[1]]$lr,
               tail(estimator$learn_rate_schedule, 1)[[1]])

  newWeights <- estimator$model$head$weight$mean()$item()

  # model should be updated when refitting
  expect_true(oldWeights != newWeights)

  directory <- withr::local_tempdir()
  estimator$save(directory, "estimator.pt")

  expect_true(file.exists(file.path(directory, "estimator.pt")))

  quietly({
    preds <- estimator$predict_proba(simDataset)
  })

  expect_lt(max(preds), 1)
  expect_gt(min(preds), 0)

  quietly({
    classes <- estimator$predict(simSmallDataset, threshold = 0.5)
  })
  expect_equal(all(unique(classes) %in% c(0, 1)), TRUE)

  quietly({
    classes <- estimator$predict(simSmallDataset$dataset)
  })
  expect_equal(all(unique(classes) %in% c(0, 1)), TRUE)

  estimatorSettings <- setEstimator(learningRate = 3e-4,
                                    weightDecay = 0.0,
                                    batchSize = 128,
                                    epochs = 5,
                                    device = "cpu",
                                    metric = "loss")
  parameters <- list(modelParameters = modelParameters,
                     estimatorSettings = estimatorSettings)
  estimator <- createEstimator(parameters = parameters)

  quietly({
    estimator$fit(simSmallDataset, simSmallDataset)
  })

  expect_equal(estimator$metric$mode, "min")
  expect_equal(estimator$metric$name, "loss")

  quietly({
    estimator$fit_whole_training_set(simSmallDataset, estimator$learn_rate_schedule)
  })

  expect_equal(estimator$learn_rate_schedule[[estimator$best_epoch]],
               estimator$optimizer$param_groups[[1]]$lr)

})

test_that("early stopping works", {
  earlyStopping <-
    importModule("Estimator")$EarlyStopping
  earlyStop <- earlyStopping(patience = 3, delta = 0, verbose = FALSE)


  testthat::expect_true(is.null(earlyStop$best_score))
  testthat::expect_false(earlyStop$early_stop)
  earlyStop(0.5)
  testthat::expect_equal(earlyStop$best_score, 0.5)
  earlyStop(0.4)
  earlyStop(0.4)
  earlyStop(0.4)
  testthat::expect_true(earlyStop$early_stop)
})

test_that("early stopping respects min_epochs", {
  earlyStopping <-
    importModule("Estimator")$EarlyStopping

  # patience exhausted before min_epochs: no stop yet
  earlyStop <- earlyStopping(patience = 1, min_epochs = 3, verbose = FALSE)
  earlyStop(0.8) # epoch 1 sets the best score
  testthat::expect_false(earlyStop$early_stop)
  earlyStop(0.7) # epoch 2: patience exhausted but grace period holds
  testthat::expect_equal(earlyStop$counter, 1)
  testthat::expect_false(earlyStop$early_stop)
  earlyStop(0.6) # epoch 3: grace period over, stop fires
  testthat::expect_true(earlyStop$early_stop)

  # default behavior unchanged when min_epochs is not set
  earlyStopDefault <- earlyStopping(patience = 2, verbose = FALSE)
  earlyStopDefault(0.8)
  earlyStopDefault(0.7)
  testthat::expect_false(earlyStopDefault$early_stop)
  earlyStopDefault(0.6)
  testthat::expect_true(earlyStopDefault$early_stop)

  # improvement during the grace period resets the counter
  earlyStopRecovery <-
    earlyStopping(patience = 1, min_epochs = 5, verbose = FALSE)
  earlyStopRecovery(0.8)
  earlyStopRecovery(0.7)
  testthat::expect_false(earlyStopRecovery$early_stop)
  earlyStopRecovery(0.9)
  testthat::expect_true(earlyStopRecovery$improved)
  testthat::expect_equal(earlyStopRecovery$counter, 0)
  testthat::expect_false(earlyStopRecovery$early_stop)
  earlyStopRecovery(0.8)
  testthat::expect_false(earlyStopRecovery$early_stop)
  earlyStopRecovery(0.7)
  testthat::expect_true(earlyStopRecovery$early_stop)
})

test_that("early stopping trains through the full minimum epoch period", {
  earlyStopping <-
    importModule("Estimator")$EarlyStopping
  earlyStop <- earlyStopping(patience = 1, min_epochs = 4, verbose = FALSE)

  for (metric in c(0.8, 0.7, 0.6)) {
    earlyStop(metric)
    testthat::expect_false(earlyStop$early_stop)
  }
  earlyStop(0.5)
  testthat::expect_true(earlyStop$early_stop)
})

test_that("early stopping with zero minimum epochs preserves default behavior", {
  earlyStopping <-
    importModule("Estimator")$EarlyStopping
  earlyStop <- earlyStopping(patience = 2, min_epochs = 0, verbose = FALSE)

  earlyStop(0.8)
  earlyStop(0.7)
  testthat::expect_false(earlyStop$early_stop)
  earlyStop(0.6)
  testthat::expect_true(earlyStop$early_stop)
})

test_that("early stopping respects minimum epochs when minimizing a metric", {
  earlyStopping <-
    importModule("Estimator")$EarlyStopping
  earlyStop <- earlyStopping(
    patience = 1, min_epochs = 3, verbose = FALSE, mode = "min"
  )

  earlyStop(0.2)
  testthat::expect_false(earlyStop$early_stop)
  earlyStop(0.3)
  testthat::expect_false(earlyStop$early_stop)
  earlyStop(0.4)
  testthat::expect_true(earlyStop$early_stop)
})

test_that("Estimator fit function works", {
  expect_true(!is.null(simFitEstimatorResults$trainDetails$trainingTime))
  expect_false(is.null(simFitEstimatorResults$trainDetails$developmentDatabase))
  expect_false(is.null(
    simFitEstimatorResults$trainDetails$developmentDatabaseSchema
  ))
  expect_false(simFitEstimatorResults$preprocessing$requiresDenseMatrix)

  expect_equal(class(simFitEstimatorResults), "plpModel")
  expect_equal(attr(simFitEstimatorResults, "modelType"), "binary")
  expect_equal(attr(simFitEstimatorResults, "saveType"), "file")
  fakeTrainData <- simTrainData
  fakeTrainData$Train$covariateData <- list(fakeCovData = c("Fake"))
  expect_error(fitEstimator(fakeTrainData$Train,
                            simModelSettings, analysisId = 1,
                            analysisPath = withr::local_tempdir()), "must be a CovariateData object", fixed = TRUE)
})

test_that("predictDeepEstimator works", {
  estimator <- fittedEstimator()

  # input is an estimator and a dataset
  # python indices are zero-based
  cohort <- simTrainData$Train$labels[unlist(simSmallDataset$indices) + 1, ]
  quietly({
    predictions <- predictDeepEstimator(estimator,
                                        simSmallDataset,
                                        cohort = cohort)
  })

  expect_lte(max(predictions$value), 1)
  expect_gte(min(predictions$value), 0)
  expect_equal(nrow(predictions), nrow(cohort))

  # input is a plpModel and data
  quietly({
    predictions <- predictDeepEstimator(
      plpModel = simFitEstimatorResults, data = simTrainData$Test,
      simTrainData$Test$labels
    )
  })
  expect_lte(max(predictions$value), 1)
  expect_gte(min(predictions$value), 0)
  expect_equal(nrow(predictions), nrow(simTrainData$Test$labels))
})

test_that("batchToDevice works", {
  estimator <- newEstimator()
  batchToDevice <- importModule("Estimator")$batch_to_device
  # since we can't test moving to gpu there is a fake device called meta
  # which we can use to test of the device is updated
  estimator$device <- "meta"
  b <- 1:10
  batch <- batchToDevice(simDataset[b], device = estimator$device)

  devices <- lapply(
    lapply(unlist(batch, recursive = TRUE), function(x) x$device),
    function(x) x == torch$device(type = "meta")
  )
  # test that all are meta
  expect_true(all(devices == TRUE))

  numDevice <- batchToDevice(simDataset[b][[1]]$feature_values, device = estimator$device)
  expect_true(numDevice$device == torch$device(type = "meta"))
})

test_that("Estimator without earlyStopping works", {
  # estimator without earlyStopping
  estimatorSettings <- setEstimator(learningRate = 3e-4,
                                    weightDecay = 0.0,
                                    batchSize = 128,
                                    epochs = 1,
                                    device = "cpu",
                                    earlyStopping = NULL)
  parameters <- list(modelParameters = modelParameters,
                     estimatorSettings = estimatorSettings)
  estimator2 <- createEstimator(parameters = parameters)
  quietly({
    estimator2$fit(simSmallDataset, simSmallDataset)
  })

  expect_null(estimator2$early_stopper)
  expect_true(!is.null(estimator2$best_epoch))

})

test_that("Early stopper can use loss and stops early", {
  estimatorSettings <- setEstimator(learningRate = 3e-2,
                                    weightDecay = 0.0,
                                    batchSize = 128,
                                    epochs = 10,
                                    device = "cpu",
                                    earlyStopping =
                                      list(useEarlyStopping = TRUE,
                                           params = list(mode = c("min"),
                                                         patience = 1)),
                                    metric = "loss",
                                    seed = 42)
  parameters <- list(modelParameters = modelParameters,
                     estimatorSettings = estimatorSettings)
  estimator <- createEstimator(parameters = parameters)
  quietly({
    estimator$fit(simSmallDataset, simSmallDataset)
  })

  expect_true(estimator$best_epoch < estimator$epochs)

})

test_that("Custom metric in estimator works", {
  skip_if_not_installed("PRROC")

  metricFun <- function(predictions, labels)  {
    pr <- PRROC::pr.curve(scores.class0 = torch$sigmoid(predictions)$numpy(),
                          weights.class0 = labels$numpy())
    auprc <- pr$auc.integral
    reticulate::r_to_py(auprc)
  }

  estimatorSettings <- setEstimator(learningRate = 3e-4,
                                    weightDecay = 0.0,
                                    batchSize = 128,
                                    device = "cpu",
                                    epochs = 1,
                                    metric = list(fun = metricFun,
                                                  name = "auprc",
                                                  mode = "max"))
  parameters <- list(modelParameters = modelParameters,
                     estimatorSettings = estimatorSettings)
  estimator <- createEstimator(parameters = parameters)
  expect_true(is.function(estimator$metric$fun))
  expect_true(is.character(estimator$metric$name))
  expect_true(estimator$metric$mode %in% c("min", "max"))

  quietly({
    estimator$fit(simSmallDataset, simSmallDataset)
  })

  expect_true(estimator$best_score[["auprc"]] > 0)

})

test_that("setEstimator with hyperparameters", {
  estimatorSettings <-
    setEstimator(learningRate = c(3e-4, 1e-3),
                 batchSize = 128,
                 epochs = 1,
                 device = "cpu",
                 metric = c("auc", "auprc"),
                 earlyStopping = list(useEarlyStopping = TRUE,
                                      params = list(patience = c(4, 10))))
  modelSettings <- setResNet(numLayers = 1, sizeHidden = 64,
                             hiddenFactor = 1, residualDropout = 0.2,
                             hiddenDropout = 0.2, sizeEmbedding = 128,
                             estimatorSettings = estimatorSettings,
                             hyperParamSearch = "grid")

  expect_true(length(modelSettings$param) == 8)
  expect_true(length(unique(lapply(modelSettings$param,
                                   function(x) x$estimator.metric))) == 2)
  expect_true(length(unique(lapply(modelSettings$param,
                                   function(x) x$estimator.learningRate))) == 2)
  expect_true(length(unique(lapply(modelSettings$param, function(x) {
    x$estimator.earlyStopping.patience
  }))) == 2)

  fitParams <-
    names(modelSettings$param[[1]])[grepl("^estimator",
                                          names(modelSettings$param[[1]]))]

  estimatorSettings2 <-
    fillEstimatorSettings(estimatorSettings, fitParams,
                          paramSearch = modelSettings$param[[8]])

  expect_equal(estimatorSettings2$learningRate, 1e-3)
  expect_equal(as.character(estimatorSettings2$metric), "auprc")
  expect_equal(estimatorSettings2$earlyStopping$params$patience, 10)
})

test_that("device as a function argument works", {
  getDevice <- function() {
    dev <- Sys.getenv("testDeepPLPDevice")
    if (dev == "") {
      dev <- "cpu"
    } else {
      dev
    }
  }

  estimatorSettings <- setEstimator(device = getDevice,
                                    learningRate = 3e-4)

  model <- setDefaultResNet(estimatorSettings = estimatorSettings)
  model$param[[1]]$feature_info <- featureInfo
  model$param[[1]]$modelType <- "ResNet"
  parameters <- list(modelParameters = model$param[[1]],
                     estimatorSettings = estimatorSettings)
  estimator <- createEstimator(parameters = parameters)

  expect_equal(estimator$device, "cpu")

  withr::local_envvar(testDeepPLPDevice = "meta")

  estimatorSettings <- setEstimator(device = getDevice,
                                    learningRate = 3e-4)

  model <- setDefaultResNet(estimatorSettings = estimatorSettings)
  model$param[[1]]$feature_info <- featureInfo
  model$param[[1]]$modelType <- "ResNet"
  parameters <- list(modelParameters = model$param[[1]],
                     estimatorSettings = estimatorSettings)
  estimator <- createEstimator(parameters = parameters)

  expect_equal(estimator$device, "meta")


})

test_that("estimatorSettings can be saved and loaded with python objects", {
  settings <- setEstimator()

  settingsFile <- withr::local_tempfile(fileext = ".RDS")
  saveRDS(settings, file = settingsFile)

  loadedSettings <- readRDS(file = settingsFile)
  optimizer <- loadedSettings$optimizer()
  scheduler <- loadedSettings$scheduler()
  criterion <- loadedSettings$criterion()

  testthat::expect_false(reticulate::py_is_null_xptr(optimizer))
  testthat::expect_false(reticulate::py_is_null_xptr(scheduler$fun))
  testthat::expect_false(reticulate::py_is_null_xptr(criterion))
})

# PatientLevelPrediction evaluates the predictions; DPLP must return them in
# the shape its evaluation expects.
test_that("predictDeepEstimator returns predictions PLP can evaluate", {
  prediction <- quietly(predictDeepEstimator(
    plpModel = simFitEstimatorResults,
    data = simTrainData$Test,
    cohort = simTrainData$Test$labels
  ))
  expect_true(all(c("rowId", "outcomeCount", "value") %in% names(prediction)))
  expect_equal(prediction$rowId, simTrainData$Test$labels$rowId)
  expect_true(all(prediction$value >= 0 & prediction$value <= 1))
  expect_equal(attr(prediction, "metaData")$modelType, "binary")
})

test_that("accumulationSteps as a function argument works", {
  getSteps <- function() {
    steps <- Sys.getenv("testAccSteps")
    if (steps == "") {
      steps <- "1"
    } else {
      steps
    }
  }

  estimatorSettings <- setEstimator(accumulationSteps = getSteps,
                                    learningRate = 3e-4,
                                    batchSize = 128)

  model <- setDefaultResNet(estimatorSettings = estimatorSettings)
  model$param[[1]]$feature_info <- featureInfo
  model$param[[1]]$modelType <- "ResNet"
  parameters <- list(modelParameters = model$param[[1]],
                     estimatorSettings = estimatorSettings)
  estimator <- createEstimator(parameters = parameters)

  expect_equal(estimator$accumulation_steps, 1)

  withr::local_envvar(testAccSteps = "4")

  estimatorSettings <- setEstimator(accumulationSteps = getSteps,
                                    learningRate = 3e-4,
                                    batchSize = 128)

  model <- setDefaultResNet(estimatorSettings = estimatorSettings)
  model$param[[1]]$feature_info <- featureInfo
  model$param[[1]]$modelType <- "ResNet"
  parameters <- list(modelParameters = model$param[[1]],
                     estimatorSettings = estimatorSettings)
  estimator <- createEstimator(parameters = parameters)

  expect_equal(estimator$accumulation_steps, 4)



})

test_that("train-validation split functionality works", {
  estimatorSettings <-
    setEstimator(learningRate = 3e-4,
                 weightDecay = 0.0,
                 batchSize = 128,
                 epochs = 1,
                 device = "cpu",
                 seed = 42,
                 optimizer = torch$optim$AdamW,
                 criterion = torch$nn$BCEWithLogitsLoss,
                 metric = "loss",
                 scheduler =
                 list(fun = torch$optim$lr_scheduler$ReduceLROnPlateau,
                      params = list(patience = 1)),
                 earlyStopping = NULL,
                 trainValidationSplit = TRUE)
  modelSettings <- setResNet(numLayers = 1, sizeHidden = 16,
                             hiddenFactor = 1, residualDropout = 0,
                             hiddenDropout = 0, sizeEmbedding = 8,
                             estimatorSettings = estimatorSettings,
                             randomSample = 1)

  fitResult <- fitEstimator(simTrainData$Train, modelSettings, analysisId = 1, analysisPath = withr::local_tempdir())

  expect_s3_class(fitResult, "plpModel")
  uniqueEvalTypes <- unique(fitResult$prediction$evaluationType)
  expect_true("Validation" %in% uniqueEvalTypes)
  expect_true("Train" %in% uniqueEvalTypes)
})
