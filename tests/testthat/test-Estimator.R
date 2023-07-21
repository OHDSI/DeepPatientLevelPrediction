Estimator <- reticulate::import_from_path("Estimator", path=path)
ResNet <- reticulate::import_from_path("ResNet", path)

catFeatures <- small_dataset$dataset$get_cat_features()$shape[[1]]
numFeatures <- small_dataset$dataset$get_numerical_features()$shape[[1]]

modelType <- ResNet$ResNet

modelParameters <- list(
  cat_features = catFeatures,
  num_features = numFeatures,
  size_embedding = 16L,
  size_hidden = 16L,
  num_layers = 2L,
  hidden_factor = 2L
)

estimatorSettings <- list(learning_rate = 3e-4,
                                 weight_decay = 0.0,
                                 batch_size = 128L,
                                 epochs = 5L,
                                 device = 'cpu',
                                 seed=42,
                                 optimizer=torch$optim$AdamW,
                                 criterion=torch$nn$BCEWithLogitsLoss,
                                 metric='loss',
                                 scheduler= list(fun=torch$optim$lr_scheduler$ReduceLROnPlateau,
                                                 params=list(patience=1)),
                                 early_stopping=NULL)
estimator <- Estimator$Estimator(model=modelType,
                                 model_parameters=modelParameters,
                                 estimator_settings=estimatorSettings)

test_that("Estimator initialization works", {

  # count parameters in both instances
  testthat::expect_equal(
    sum(reticulate::iterate(estimator$model$parameters(), function(x) x$numel())),
    sum(reticulate::iterate(do.call(modelType, modelParameters)$parameters(), 
                            function(x) x$numel()))
  )

  testthat::expect_equal(
    estimator$model_parameters,
    modelParameters
  )

  
})
sink(nullfile())
estimator$fit(small_dataset, small_dataset)
sink()

test_that("estimator fitting works", {

  expect_true(!is.null(estimator$best_epoch))
  expect_true(!is.null(estimator$best_score$loss))
  expect_true(!is.null(estimator$best_score$auc))

  old_weights <- estimator$model$head$weight$mean()$item()

  sink(nullfile())
  estimator$fit_whole_training_set(small_dataset, estimator$learn_rate_schedule)
  sink()

  expect_equal(estimator$optimizer$param_groups[[1]]$lr, tail(estimator$learn_rate_schedule, 1)[[1]])

  new_weights <- estimator$model$head$weight$mean()$item()

  # model should be updated when refitting
  expect_true(old_weights != new_weights)

  estimator$save(testLoc, "estimator.pt")

  expect_true(file.exists(file.path(testLoc, "estimator.pt")))
  
  sink(nullfile())
  preds <- estimator$predict_proba(dataset)
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
                                    batchSize = 128L,
                                    epochs = 5L,
                                    device = 'cpu',
                                    metric= "loss")
  estimator_settings <- camelCaseToSnakeCaseNames(estimatorSettings)
  estimator <- Estimator$Estimator(model=modelType,
                                   model_parameters=modelParameters,
                                   estimator_settings=estimator_settings)
  
  sink(nullfile())
  estimator$fit(small_dataset, small_dataset)
  sink()
  
  expect_equal(estimator$metric$mode, "min")
  expect_equal(estimator$metric$name, "loss")
  
  sink(nullfile())
  estimator$fit_whole_training_set(small_dataset, estimator$learn_rate_schedule)
  sink()
  
  expect_equal(estimator$learn_rate_schedule[[estimator$best_epoch]],
               estimator$optimizer$param_groups[[1]]$lr)
  
})

test_that("early stopping works", {
  EarlyStopping <- reticulate::import_from_path("Estimator", path=path)$EarlyStopping
  earlyStop <- EarlyStopping(patience = 3, delta = 0, verbose = FALSE)
  
  
  testthat::expect_true(is.null(earlyStop$best_score))
  testthat::expect_false(earlyStop$early_stop)
  earlyStop(0.5)
  testthat::expect_equal(earlyStop$best_score, 0.5)
  earlyStop(0.4)
  earlyStop(0.4)
  earlyStop(0.4)
  testthat::expect_true(earlyStop$early_stop)
})

modelSettings <- setResNet(
  numLayers = 1L, sizeHidden = 16L, hiddenFactor = 1L,
  residualDropout = 0, hiddenDropout = 0,
  sizeEmbedding = 16L, hyperParamSearch = "random",
  randomSample = 1L,
  setEstimator(epochs=1L,
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
  batch_to_device <- reticulate::import_from_path("Estimator", path=path)$batch_to_device
  # since we can't test moving to gpu there is a fake device called meta
  # which we can use to test of the device is updated
  estimator$device <- "meta"
  b <- 1:10
  batch <- batch_to_device(dataset[b], device=estimator$device)
  
  devices <- lapply(
    lapply(unlist(batch, recursive = TRUE), function(x) x$device),
    function(x) x == torch$device(type = "meta")
  )
  # test that all are meta
  expect_true(all(devices == TRUE))
  
  numDevice <- batch_to_device(dataset[b][[1]]$num, device=estimator$device)
  expect_true(numDevice$device==torch$device(type="meta"))
})

test_that("Estimator without earlyStopping works", {
  # estimator without earlyStopping
  estimatorSettings <- setEstimator(learningRate = 3e-4,
                                    weightDecay = 0.0,
                                    batchSize = 128L,
                                    epochs = 1L,
                                    device = 'cpu',
                                    earlyStopping = NULL)
  
  estimator2 <- Estimator$Estimator(model = modelType,
                                    model_parameters = modelParameters,
                                    estimator_settings = estimatorSettings)
  
  sink(nullfile())
  estimator2$fit(small_dataset, small_dataset)
  sink()
  
  expect_null(estimator2$early_stopper)
  expect_true(!is.null(estimator2$best_epoch))
  
})

test_that("Early stopper can use loss and stops early", {
  estimatorSettings <- setEstimator(learningRate = 3e-2,
                                    weightDecay = 0.0,
                                    batchSize = 128L,
                                    epochs = 10L,
                                    device = 'cpu',
                                    earlyStopping =list(useEarlyStopping=TRUE,
                                                        params = list(mode=c('min'), 
                                                                      patience=1L)),
                                    metric = 'loss',
                                    seed=42)
  estimator_settings <- snakeCaseToCamelCaseNames(estimatorSettings)
  estimator <- Estimator$Estimator(model = modelType,
                                    model_parameters = modelParameters,
                                    estimator_settings = estimator_settings)
  sink(nullfile())
  estimator$fit(small_dataset, small_dataset)
  sink()
  
  expect_true(estimator$best_epoch < estimator$epochs)
  
})

test_that('Custom metric in estimator works', {
  
  metric_fun <- function(predictions, labels)  {
    pr <- PRROC::pr.curve(scores.class0 = torch$sigmoid(predictions)$numpy(), 
                          weights.class0 = labels$numpy())
    auprc <- pr$auc.integral
    reticulate::r_to_py(auprc)
  }
    
  estimatorSettings <- setEstimator(learningRate = 3e-4,
                                    weightDecay = 0.0,
                                    batchSize = 128L,
                                    device = "cpu",
                                    epochs = 1L,
                                    metric=list(fun=metric_fun,
                                                name="auprc",
                                                mode="max"))
  estimator_settings <- camelCaseToSnakeCaseNames(estimatorSettings)
  estimator <- Estimator$Estimator(model = modelType,
                                   model_parameters = modelParameters,
                                   estimator_settings = estimator_settings)
  expect_true(is.function(estimator$metric$fun))
  expect_true(is.character(estimator$metric$name))
  expect_true(estimator$metric$mode %in% c("min", "max"))
  
  sink(nullfile())
  estimator$fit(small_dataset, small_dataset)
  sink()
  
  expect_true(estimator$best_score[["auprc"]]>0)

})

test_that("setEstimator with paramsToTune is correctly added to hyperparameters", {
  estimatorSettings <- setEstimator(learningRate=c(3e-4,1e-3),
                                    batchSize=128L,
                                    epochs=1L,
                                    device="cpu",
                                    metric=c("auc", "auprc"),
                                    earlyStopping = list(useEarlyStopping=TRUE,
                                                         params=list(patience=c(4L,10L))))
  modelSettings <- setResNet(numLayers = 1L, sizeHidden = 64L,
                             hiddenFactor = 1L, residualDropout = 0.2,
                             hiddenDropout = 0.2,sizeEmbedding = 128L,
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

test_that("device as a function argument works", {
  getDevice <- function() {
    dev <- Sys.getenv("testDeepPLPDevice") 
    if (dev == ""){
      dev = "cpu"
    } else{
      dev
    }
  }

  estimatorSettings <- setEstimator(device=getDevice,
                                    learningRate = 3e-4)
  
  model <- setDefaultResNet(estimatorSettings = estimatorSettings) 
  model$param[[1]]$catFeatures <- 10L
  
  model_parameters <- camelCaseToSnakeCaseNames(model$param[[1]])
  estimator_settings <- camelCaseToSnakeCaseNames(estimatorSettings)
  estimator <- Estimator$Estimator(model = modelType,
                             model_parameters = model_parameters,
                             estimator_settings = estimator_settings)
  
  expect_equal(estimator$device, "cpu")
  
  Sys.setenv("testDeepPLPDevice" = "meta")
  
  estimatorSettings <- setEstimator(device=getDevice,
                                    learningRate = 3e-4)
  
  model <- setDefaultResNet(estimatorSettings = estimatorSettings) 
  model$param[[1]]$catFeatures <- 10L
  
  model_parameters <- camelCaseToSnakeCaseNames(model$param[[1]])
  estimator_settings <- camelCaseToSnakeCaseNames(estimatorSettings)
  estimator <- Estimator$Estimator(model = modelType,
                                   model_parameters = model_parameters,
                                   estimator_settings = estimator_settings)
  
  expect_equal(estimator$device, "meta")
  
  Sys.unsetenv("testDeepPLPDevice")
  
  })
