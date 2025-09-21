featureInfo <- smallDataset$dataset$get_feature_info()
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
estimator <- createEstimator(parameters = parameters)

test_that("Estimator initialization works", {

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

  testthat::expect_error(setEstimator(learningRate = "notAuto"))
  testthat::expect_error(setEstimator(weightDecay = -1))
  testthat::expect_error(setEstimator(weightDecay = "text"))
  testthat::expect_error(setEstimator(batchSize = 0))
  testthat::expect_error(setEstimator(batchSize = "text"))
  testthat::expect_error(setEstimator(epochs = 0))
  testthat::expect_error(setEstimator(epochs = "test"))
  testthat::expect_error(setEstimator(earlyStopping = "notListorNull"))
  testthat::expect_error(setEstimator(metric = 1))
  testthat::expect_error(setEstimator(seed = "32"))
})

sink(nullfile())
estimator$fit(smallDataset, smallDataset)
sink()

test_that("estimator fitting works", {

  expect_true(!is.null(estimator$best_epoch))
  expect_true(!is.null(estimator$best_score$loss))
  expect_true(!is.null(estimator$best_score$auc))

  oldWeights <- estimator$model$head$weight$mean()$item()

  sink(nullfile())
  estimator$fit_whole_training_set(smallDataset, estimator$learn_rate_schedule)
  sink()

  expect_equal(estimator$optimizer$param_groups[[1]]$lr,
               tail(estimator$learn_rate_schedule, 1)[[1]])

  newWeights <- estimator$model$head$weight$mean()$item()

  # model should be updated when refitting
  expect_true(oldWeights != newWeights)

  estimator$save(testLoc, "estimator.pt")

  expect_true(file.exists(file.path(testLoc, "estimator.pt")))

  sink(nullfile())
  preds <- estimator$predict_proba(dataset)
  sink()

  expect_lt(max(preds), 1)
  expect_gt(min(preds), 0)

  sink(nullfile())
  classes <- estimator$predict(smallDataset, threshold = 0.5)
  sink()
  expect_equal(all(unique(classes) %in% c(0, 1)), TRUE)

  sink(nullfile())
  classes <- estimator$predict(smallDataset$dataset)
  sink()
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

  sink(nullfile())
  estimator$fit(smallDataset, smallDataset)
  sink()

  expect_equal(estimator$metric$mode, "min")
  expect_equal(estimator$metric$name, "loss")

  sink(nullfile())
  estimator$fit_whole_training_set(smallDataset, estimator$learn_rate_schedule)
  sink()

  expect_equal(estimator$learn_rate_schedule[[estimator$best_epoch]],
               estimator$optimizer$param_groups[[1]]$lr)

})

test_that("early stopping works", {
  earlyStopping <-
    reticulate::import_from_path("Estimator", path = path)$EarlyStopping
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

test_that("Estimator fit function works", {
  expect_true(!is.null(fitEstimatorResults$trainDetails$trainingTime))

  expect_equal(class(fitEstimatorResults), "plpModel")
  expect_equal(attr(fitEstimatorResults, "modelType"), "binary")
  expect_equal(attr(fitEstimatorResults, "saveType"), "file")
  fakeTrainData <- trainData
  fakeTrainData$train$covariateData <- list(fakeCovData = c("Fake"))
  expect_error(fitEstimator(fakeTrainData$train,
                            modelSettings, analysisId = 1,
                            analysisPath = testLoc))
})

test_that("predictDeepEstimator works", {

  # input is an estimator and a dataset
  sink(nullfile())
  predictions <- predictDeepEstimator(estimator,
                                      smallDataset,
                                      cohort = trainData$Train$labels)
  sink()

  expect_lte(max(predictions$value), 1)
  expect_gte(min(predictions$value), 0)
  expect_equal(nrow(predictions), nrow(trainData$Train$labels))

  # input is a plpModel and data
  sink(nullfile())
  predictions <- predictDeepEstimator(
    plpModel = fitEstimatorResults, data = trainData$Test,
    trainData$Test$labels
  )
  sink()
  expect_lte(max(predictions$value), 1)
  expect_gte(min(predictions$value), 0)
  expect_equal(nrow(predictions), nrow(trainData$Test$labels))
})

test_that("batchToDevice works", {
  batchToDevice <- reticulate::import_from_path("Estimator",
                                                path = path)$batch_to_device
  # since we can't test moving to gpu there is a fake device called meta
  # which we can use to test of the device is updated
  estimator$device <- "meta"
  b <- 1:10
  batch <- batchToDevice(dataset[b], device = estimator$device)

  devices <- lapply(
    lapply(unlist(batch, recursive = TRUE), function(x) x$device),
    function(x) x == torch$device(type = "meta")
  )
  # test that all are meta
  expect_true(all(devices == TRUE))

  numDevice <- batchToDevice(dataset[b][[1]]$feature_values, device = estimator$device)
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
  sink(nullfile())
  estimator2$fit(smallDataset, smallDataset)
  sink()

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
  sink(nullfile())
  estimator$fit(smallDataset, smallDataset)
  sink()

  expect_true(estimator$best_epoch < estimator$epochs)

})

test_that("Custom metric in estimator works", {

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

  sink(nullfile())
  estimator$fit(smallDataset, smallDataset)
  sink()

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

  Sys.setenv("testDeepPLPDevice" = "meta")

  estimatorSettings <- setEstimator(device = getDevice,
                                    learningRate = 3e-4)

  model <- setDefaultResNet(estimatorSettings = estimatorSettings)
  model$param[[1]]$feature_info <- featureInfo
  model$param[[1]]$modelType <- "ResNet"
  parameters <- list(modelParameters = model$param[[1]],
                     estimatorSettings = estimatorSettings)
  estimator <- createEstimator(parameters = parameters)

  expect_equal(estimator$device, "meta")

  Sys.unsetenv("testDeepPLPDevice")

})

test_that("estimatorSettings can be saved and loaded with python objects", {
  settings <- setEstimator()

  saveRDS(settings, file = file.path(testLoc, "settings.RDS"))

  loadedSettings <- readRDS(file = file.path(testLoc, "settings.RDS"))
  optimizer <- loadedSettings$optimizer()
  scheduler <- loadedSettings$scheduler()
  criterion <- loadedSettings$criterion()

  testthat::expect_false(reticulate::py_is_null_xptr(optimizer))
  testthat::expect_false(reticulate::py_is_null_xptr(scheduler$fun))
  testthat::expect_false(reticulate::py_is_null_xptr(criterion))
})

test_that("evaluation works on predictDeepEstimator output", {
  
  prediction <- predictDeepEstimator(plpModel = fitEstimatorResults,
                                     data = trainData$Test,
                                     cohort = trainData$Test$labels)
  prediction$evaluationType <- 'Validation'

  evaluation <-
    PatientLevelPrediction::evaluatePlp(prediction, "evaluationType")
  
  expect_length(evaluation, 5)
  expect_s3_class(evaluation, "plpEvaluation")
  
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

  Sys.setenv("testAccSteps" = "4")

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


  Sys.unsetenv("testAccSteps")

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

  fitResult <- fitEstimator(trainData$Train, modelSettings, analysisId = 1, analysisPath = tempdir())

  expect_s3_class(fitResult, "plpModel")
  uniqueEvalTypes <- unique(fitResult$prediction$evaluationType)
  expect_true("Validation" %in% uniqueEvalTypes)
  expect_true("Train" %in% uniqueEvalTypes)
})

test_that("setEstimator errors on invalid precision", {
  expect_error(
    setEstimator(precision = "fp32"),
    regexp = "precision must be one of 'float32', 'float16', 'bfloat16'",
    fixed = FALSE
  )
})
