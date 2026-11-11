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

test_that("Estimator tie-breaking chooses the last best epoch", {
  estimatorModule <- importModule("Estimator")
  expect_equal(estimatorModule$select_best_epoch(list(0.1, 0.5, 0.5), "max"), 2L)
  expect_equal(estimatorModule$select_best_epoch(list(0.2, 0.1, 0.1), "min"), 2L)
})

test_that("the restored checkpoint holds the weights of the best epoch", {
  skip_if_no_python()
  toy <- toyPython()
  estimatorClass <- importModule("Estimator")$Estimator
  dataset <- toy$ToyData(c(-1, 1), c(1, 0))

  # the weight moves away from the optimum, so the first epoch is the best one
  earlyStopping <- list(fun = NULL, params = list(patience = 1L, mode = "min"))
  for (settings in list(list(early_stopping = NULL), list(early_stopping = earlyStopping))) {
    estimator <- do.call(toy$toy_estimator, c(
      list(estimatorClass, epochs = 4L, learning_rate = 0.5, batch_size = 2L),
      settings
    ))
    epochScores <- c(1, 2, 3, 4)
    epoch <- 0L
    weightsPerEpoch <- numeric()
    reticulate::py_set_attr(estimator, "score", function(dataloader) {
      epoch <<- epoch + 1L
      weightsPerEpoch[epoch] <<- estimator$model$weight$item()
      list(metric = epochScores[epoch], auc = 0.5, loss = epochScores[epoch])
    })
    estimator$fit(dataset, dataset)

    expect_equal(estimator$best_epoch, 0L)
    expect_equal(estimator$model$weight$item(), weightsPerEpoch[1], tolerance = 1e-6)
    expect_false(isTRUE(all.equal(weightsPerEpoch[1], weightsPerEpoch[2])))
  }
})

test_that("checkpoints do not accumulate over the epochs", {
  skip_if_no_python()
  toy <- toyPython()
  estimator <- toy$toy_estimator(
    importModule("Estimator")$Estimator, early_stopping = NULL
  )
  checkpoints <- reticulate::dict()
  scores <- list(list(metric = 3), list(metric = 2), list(metric = 1))
  for (epoch in 0:2) {
    estimator$`_keep_best_checkpoint`(checkpoints, scores, epoch)
  }
  expect_equal(reticulate::py_len(checkpoints), 1L)
  expect_false(is.null(checkpoints$get(2L)))
})

biasPython <- function() {
  toyPython()
  reticulate::py_run_string("
class HeadModel(nn.Module):
    name = 'HeadModel'
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1)
    def forward(self, x):
        return self.head(x['feature_values']).squeeze(-1)

class WrappedHeadModel(nn.Module):
    name = 'WrappedHeadModel'
    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.head.linear(x['feature_values']).squeeze(-1)

class CustomLossModel(HeadModel):
    def compute_loss(self, predictions, targets, criterion):
        return criterion(predictions.squeeze(), targets.float())

class LabelledData(ToyData):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.target = self.y

def bias_estimator(estimator_class, model, **settings):
    # a learning rate of zero keeps the initial bias through training
    defaults = dict(device='cpu', find_l_r=False, learning_rate=0.0, weight_decay=0.0,
                    batch_size=4, epochs=1, optimizer=torch.optim.SGD,
                    criterion=torch.nn.BCEWithLogitsLoss, metric='loss',
                    init_output_bias=True)
    return estimator_class(model, {'model_parameters': {},
                                   'estimator_settings': {**defaults, **settings}})
")
}

test_that("the output bias starts at the outcome rate of the training data", {
  skip_if_no_python()
  py <- biasPython()
  Estimator <- importModule("Estimator")$Estimator
  y <- c(1, 0, 0, 0, 0, 0, 0, 1)
  data <- py$LabelledData(seq_along(y), y)

  estimator <- py$bias_estimator(Estimator, py$HeadModel)
  estimator$fit_whole_training_set(data, list(0))
  expect_equal(asNumeric(estimator$model$head$bias), qlogis(2 / 8), tolerance = 1e-6)

  # a cross-validation fit uses the outcome rate of its training fold only
  estimator <- py$bias_estimator(Estimator, py$HeadModel)
  estimator$fit(
    torch$utils$data$Subset(data, indices = 0:3),
    torch$utils$data$Subset(data, indices = 4:7)
  )
  expect_equal(asNumeric(estimator$model$head$bias), qlogis(1 / 4), tolerance = 1e-6)

  # transformer heads keep their linear layer in head.linear
  estimator <- py$bias_estimator(Estimator, py$WrappedHeadModel)
  estimator$fit_whole_training_set(data, list(0))
  expect_equal(asNumeric(estimator$model$head$linear$bias), qlogis(2 / 8), tolerance = 1e-6)
})

test_that("the output bias is left alone when switched off or the model has its own loss", {
  skip_if_no_python()
  py <- biasPython()
  Estimator <- importModule("Estimator")$Estimator
  y <- c(1, 0, 0, 0, 0, 0, 0, 1)
  data <- py$LabelledData(seq_along(y), y)
  cases <- list(
    list(model = py$HeadModel, init = FALSE),
    list(model = py$CustomLossModel, init = TRUE)
  )
  for (case in cases) {
    estimator <- py$bias_estimator(Estimator, case$model, init_output_bias = case$init)
    before <- asNumeric(estimator$model$head$bias)
    estimator$fit_whole_training_set(data, list(0))
    expect_equal(asNumeric(estimator$model$head$bias), before)
  }
})

test_that("useEarlyStopping = FALSE switches early stopping off", {
  skip_if_no_python()
  py <- toyPython()
  Estimator <- importModule("Estimator")$Estimator
  earlyStopper <- function(useEarlyStopping) {
    estimator <- py$toy_estimator(
      Estimator,
      early_stopping = list(useEarlyStopping = useEarlyStopping, params = list(patience = 1L))
    )
    estimator$early_stopper
  }
  expect_false(is.null(earlyStopper(TRUE)))
  expect_null(earlyStopper(FALSE))
})

test_that("gradient clipping limits the gradient norm of each optimizer step", {
  skip_if_no_python()
  py <- toyPython()
  Estimator <- importModule("Estimator")$Estimator
  y <- c(1, 0, 0, 0, 0, 0, 0, 1)
  data <- py$ToyData(seq_along(y), y)
  weightAfterOneStep <- function(...) {
    estimator <- py$toy_estimator(Estimator, batch_size = 8L, ...)
    estimator$fit_whole_training_set(data, list(1.0))
    estimator$model$weight$item()
  }
  # at weight 0 the gradient of the summed loss is sum((0.5 - y) * x) = 9, and
  # SGD with a learning rate of 1 subtracts it
  expect_equal(weightAfterOneStep(), -9, tolerance = 1e-6)
  expect_equal(weightAfterOneStep(grad_clip_norm = 0.5), -0.5, tolerance = 1e-5)
})

test_that("each estimator sets deterministic training on or off", {
  skip_if_no_python()
  py <- toyPython()
  Estimator <- importModule("Estimator")$Estimator
  os <- reticulate::import("os")
  previousConfig <- os$environ$get("CUBLAS_WORKSPACE_CONFIG")
  withr::defer({
    torch$use_deterministic_algorithms(FALSE)
    torch$backends$cudnn$deterministic <- FALSE
    if (is.null(previousConfig)) os$environ$pop("CUBLAS_WORKSPACE_CONFIG", NULL)
  })

  py$toy_estimator(Estimator, deterministic = TRUE)
  expect_true(torch$are_deterministic_algorithms_enabled())
  expect_true(torch$is_deterministic_algorithms_warn_only_enabled())
  expect_true(torch$backends$cudnn$deterministic)
  expect_false(is.null(os$environ$get("CUBLAS_WORKSPACE_CONFIG")))

  # the setting is for the whole process, so the next fit must not inherit it
  py$toy_estimator(Estimator)
  expect_false(torch$are_deterministic_algorithms_enabled())
  expect_false(torch$backends$cudnn$deterministic)
})

test_that("AUC gives half credit to tied scores", {
  skip_if_no_python()
  computeAuc <- importModule("Estimator")$compute_auc
  auc <- function(labels, scores) {
    computeAuc(
      torch$tensor(labels, dtype = torch$float32),
      torch$tensor(scores, dtype = torch$float32)
    )$item()
  }
  expect_equal(auc(c(1, 0), c(0, 0)), 0.5)
  expect_equal(auc(c(0, 1), c(0, 0)), 0.5)
  expect_equal(auc(c(1, 0), c(1, 0)), 1)
  expect_equal(auc(c(1, 0), c(0, 1)), 0)
  # pairs: (0.9>0.5), (0.9>0.1), (0.5=0.5 tie), (0.5>0.1)
  expect_equal(auc(c(1, 1, 0, 0), c(0.9, 0.5, 0.5, 0.1)), 3.5 / 4)

  set.seed(1)
  labels <- rep(c(0, 1), 50)
  scores <- sample(c(0.1, 0.2, 0.3), 100, replace = TRUE)
  prediction <- data.frame(outcomeCount = labels, value = scores)
  expect_equal(
    auc(labels, scores),
    PatientLevelPrediction::computeAuc(prediction),
    tolerance = 1e-6
  )
})

test_that("validation works when its last batch has a single patient", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  # folds of four patients scored in batches of three leave one patient
  settings <- setMultiLayerPerceptron(
    numLayers = 1, sizeHidden = 4, dropout = 0, sizeEmbedding = 4,
    estimatorSettings = setEstimator(
      learningRate = 0.01, batchSize = 3, epochs = 1, device = "cpu", seed = 42
    ),
    randomSample = 1
  )
  result <- gridCvDeep(
    mappedData = data, labels = createMockLabels(), modelSettings = settings,
    modelLocation = withr::local_tempdir(), analysisPath = withr::local_tempdir()
  )
  cv <- result$prediction[result$prediction$evaluationType == "CV", ]
  expect_equal(nrow(cv), 8)
})

test_that("final fitting trains when the data is smaller than one batch", {
  skip_if_no_python()
  toy <- toyPython()
  estimator <- toy$toy_estimator(
    importModule("Estimator")$Estimator, batch_size = 512L, epochs = 1L
  )
  estimator$fit_whole_training_set(toy$ToyData(c(-1, 1), c(0, 1)))
  expect_false(estimator$model$weight$item() == 0)

  emptyEstimator <- toy$toy_estimator(importModule("Estimator")$Estimator)
  expect_error(
    emptyEstimator$fit_whole_training_set(toy$ToyData(numeric(), numeric())),
    "empty dataset"
  )
})

test_that("reported losses are averaged over the actual observations", {
  skip_if_no_python()
  toy <- toyPython()
  dataLoader <- torch$utils$data$DataLoader
  sampler <- torch$utils$data$BatchSampler
  sequential <- torch$utils$data$SequentialSampler
  estimator <- toy$toy_estimator(
    importModule("Estimator")$Estimator, batch_size = 8L
  )
  # a batch of 8 and a partial batch of 2, with a zero weight every loss is log(2)
  dataset <- toy$ToyData(rep(1, 10), rep(c(0, 1), 5))
  loader <- dataLoader(
    dataset = dataset, batch_size = NULL,
    sampler = sampler(sequential(dataset), batch_size = 8L, drop_last = FALSE)
  )
  expect_equal(estimator$score(loader)$loss, log(2), tolerance = 1e-6)
  expect_equal(estimator$fit_epoch(loader), log(2), tolerance = 1e-6)
})
