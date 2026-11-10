# Regression tests for model selection, training and feature handling bugs.
pythonPath <- system.file("python", package = "DeepPatientLevelPrediction")

importModule <- function(module) {
  reticulate::import_from_path(module, path = pythonPath)
}

toyPython <- function() {
  reticulate::py_run_string("
import torch
from torch import nn
from torch.utils.data import Dataset

class ToyData(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, item):
        index = torch.as_tensor(item, dtype=torch.long).reshape(-1)
        return [{'feature_ids': self.x[index].long(), 'feature_values': self.x[index]},
                self.y[index].reshape(-1)]

class ToyModel(nn.Module):
    name = 'ToyModel'
    def __init__(self, weight=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(weight)))
    def forward(self, x):
        return (self.weight * x['feature_values']).squeeze(-1)

def toy_estimator(estimator_class, **settings):
    defaults = dict(device='cpu', find_l_r=False, learning_rate=0.1, batch_size=2,
                    epochs=3, optimizer=torch.optim.SGD,
                    criterion=torch.nn.BCEWithLogitsLoss, metric='loss')
    return estimator_class(ToyModel, {'model_parameters': {},
                                      'estimator_settings': {**defaults, **settings}})
")
}

asNumeric <- function(tensor) {
  as.numeric(tensor$detach()$cpu()$numpy())
}

createMockData <- function(
    covariates = data.frame(
      rowId = rep(1:8, each = 2),
      covariateId = rep(c(101, 102), 8),
      columnId = rep(1:2, 8),
      covariateValue = rep(c(1, 2), 8)
    ),
    isBinary = "N",
    timeRef = NULL) {
  columnIds <- sort(unique(covariates$columnId))
  tables <- list(
    analysisRef = data.frame(analysisId = 1, isBinary = isBinary, missingMeansZero = "N"),
    covariateRef = data.frame(
      columnId = columnIds,
      covariateId = columnIds + 100,
      analysisId = 1,
      conceptId = columnIds + 1000,
      covariateName = paste("covariate", columnIds)
    ),
    covariates = covariates
  )
  if (!is.null(timeRef)) {
    tables$timeRef <- timeRef
  }
  do.call(Andromeda::andromeda, tables)
}

createMockLabels <- function(n = 8) {
  data.frame(
    rowId = seq_len(n),
    outcomeCount = rep(c(0, 1), length.out = n),
    index = rep(c(1, 1, 2, 2), length.out = n),
    cohortStartDate = as.Date("2020-01-01")
  )
}

createMockSettings <- function(learningRate = 0.01, seed = 42, ...) {
  setMultiLayerPerceptron(
    numLayers = 1, sizeHidden = 4, dropout = 0, sizeEmbedding = 4,
    estimatorSettings = setEstimator(
      learningRate = learningRate, batchSize = 4, epochs = 1, device = "cpu",
      seed = seed, ...
    ),
    randomSample = 1
  )
}

mappedFeatureInfo <- function(covariateIds = 1:4 + 100, conceptIds = 1:4 + 1000) {
  polars <- reticulate::import("polars", convert = FALSE)
  columnIds <- seq_along(covariateIds)
  importModule("Dataset")$FeatureInfo(polars$DataFrame(list(
    columnId = columnIds,
    covariateId = covariateIds,
    conceptId = conceptIds,
    isBinary = rep("Y", length(columnIds))
  )))
}

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
    def compute_loss(self, predictions, targets, criterion, label_smoothing=0.0):
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

  # the loss sees smoothed labels, so their rate is used
  estimator <- py$bias_estimator(Estimator, py$HeadModel, label_smoothing = 0.1)
  estimator$fit_whole_training_set(data, list(0))
  expect_equal(asNumeric(estimator$model$head$bias), qlogis(2 / 8 * 0.9 + 0.05), tolerance = 1e-6)

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

test_that("setEstimator turns on the output bias initialization by default", {
  settings <- setEstimator(seed = 1)
  expect_true(settings$initOutputBias)
  expect_false(setEstimator(seed = 1, initOutputBias = FALSE)$initOutputBias)
  expect_error(setEstimator(seed = 1, initOutputBias = "yes"))
  expect_named(camelCaseToSnakeCaseNames(list(initOutputBias = TRUE)), "init_output_bias")
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

test_that("RealMLP passes its beta2 setting to the optimizer", {
  skip_if_no_python()
  settings <- setRealMLP(sizeHidden = 4L, numLayers = 1L, sizeEmbedding = 4L)
  parameters <- settings$param[[1]]
  parameters$modelType <- settings$modelType
  parameters$feature_info <- mappedFeatureInfo()
  estimator <- createEstimator(list(
    modelParameters = parameters,
    estimatorSettings = settings$estimatorSettings
  ))
  expect_equal(estimator$beta2, settings$estimatorSettings$beta2)
  for (group in estimator$optimizer$param_groups) {
    expect_equal(group$betas[[2]], settings$estimatorSettings$beta2)
  }
})

customEmbeddingEstimator <- function(concepts, weights, conceptIds = 1:4 + 1000) {
  embeddingFile <- withr::local_tempfile(fileext = ".pt", .local_envir = parent.frame())
  torch$save(list(
    concept_ids = torch$tensor(as.array(as.integer(concepts)), dtype = torch$long),
    embeddings = torch$tensor(weights, dtype = torch$float)
  ), embeddingFile)
  settings <- setCustomEmbeddingModel(
    embeddingFilePath = embeddingFile,
    modelSettings = setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 4, dropout = 0, sizeEmbedding = 2,
      estimatorSettings = setEstimator(learningRate = 0.001, device = "cpu", seed = 42),
      randomSample = 1
    )
  )
  parameters <- settings$param[[1]]
  parameters$modelType <- settings$modelType
  parameters$feature_info <- mappedFeatureInfo(conceptIds = conceptIds)
  createEstimator(list(
    modelParameters = parameters,
    estimatorSettings = settings$estimatorSettings
  ))
}

embedRow <- function(estimator, ids) {
  ids <- matrix(as.integer(ids), nrow = 1)
  asNumeric(estimator$model$embedding(list(
    feature_ids = torch$tensor(ids, dtype = torch$long),
    feature_values = torch$ones(dim(ids), dtype = torch$float)
  )))
}

test_that("custom embeddings are assigned to their own covariates", {
  skip_if_no_python()
  # concepts in the file are in the opposite order of their columns
  estimator <- customEmbeddingEstimator(
    concepts = c(1003, 1001),
    weights = rbind(c(3, 30), c(1, 10))
  )
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), c(3L, 1L))
  expect_equal(embedRow(estimator, 1), c(1, 10))
  expect_equal(embedRow(estimator, 3), c(3, 30))
})

test_that("custom embeddings work for a single matching concept", {
  skip_if_no_python()
  estimator <- customEmbeddingEstimator(concepts = 1002, weights = rbind(c(5, 50)))
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), 2L)
  expect_equal(embedRow(estimator, 2), c(5, 50))
})

test_that("custom embedding covariates sharing a concept each get its vector", {
  skip_if_no_python()
  estimator <- customEmbeddingEstimator(
    concepts = 1001,
    weights = rbind(c(7, 70)),
    conceptIds = c(1001, 1002, 1001, 1003)
  )
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), c(1L, 3L))
  expect_equal(embedRow(estimator, 1), c(7, 70))
  expect_equal(embedRow(estimator, 3), c(7, 70))
})

test_that("a concept with several vectors in the embeddings file is an error", {
  skip_if_no_python()
  expect_error(
    customEmbeddingEstimator(concepts = c(1002, 1003, 1002), weights = rbind(c(1, 2), c(3, 4), c(5, 6))),
    "more than one vector for concept IDs 1002"
  )
  # a repeat that is not in the training data is dropped like other unused concepts
  estimator <- customEmbeddingEstimator(
    concepts = c(1002, 9999, 9999), weights = rbind(c(1, 2), c(3, 4), c(5, 6))
  )
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), 2L)
})

test_that("aggregated custom embeddings do not depend on padding", {
  skip_if_no_python()
  estimator <- customEmbeddingEstimator(concepts = 1001, weights = rbind(c(1, 10)))
  # one custom covariate with one regular covariate
  unpadded <- embedRow(estimator, c(1, 2))
  expect_equal(embedRow(estimator, c(1, 2, 0, 0)), unpadded, tolerance = 1e-6)
  regular <- embedRow(estimator, 2)
  expect_equal(unpadded, (c(1, 10) + regular) / 2, tolerance = 1e-6)
  expect_equal(embedRow(estimator, c(1, 0, 0)), c(1, 10), tolerance = 1e-6)
})

test_that("predictions are kept for patients without covariates", {
  skip_if_no_python()
  data <- createMockData(data.frame(
    rowId = c(1, 1, 3),
    covariateId = c(101, 102, 101),
    columnId = c(1L, 2L, 1L),
    covariateValue = c(1, 1, 2)
  ))
  withr::defer(Andromeda::close(data))
  plpModel <- list(covariateImportance = data.frame(
    columnId = 1:2, covariateId = c(101, 102), analysisId = 1,
    conceptId = c(1001, 1002), covariateName = c("a", "b")
  ))
  dataset <- createDataset(data, plpModel = plpModel, numRows = 4)

  expect_equal(reticulate::py_len(dataset), 4L)
  featureIds <- dataset$data$feature_ids$numpy()
  expect_equal(unname(featureIds[1, ]), c(1, 2))
  expect_equal(unname(featureIds[2, ]), c(0, 0))
  expect_equal(unname(featureIds[3, ]), c(1, 0))
  expect_equal(unname(featureIds[4, ]), c(0, 0))

  settingsList <- list(
    setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 4, dropout = 0, sizeEmbedding = 4,
      estimatorSettings = setEstimator(learningRate = 0.01, device = "cpu", seed = 42),
      randomSample = 1
    ),
    setResNet(
      numLayers = 1, sizeHidden = 4, hiddenFactor = 1, residualDropout = 0,
      hiddenDropout = 0, sizeEmbedding = 4,
      estimatorSettings = setEstimator(learningRate = 0.01, device = "cpu", seed = 42),
      randomSample = 1
    ),
    setTransformer(
      numBlocks = 1, dimToken = 4, dimOut = 1, numHeads = 2, attDropout = 0,
      ffnDropout = 0, dimHidden = 4,
      estimatorSettings = setEstimator(learningRate = 0.01, device = "cpu", seed = 42),
      randomSample = 1
    ),
    setRealMLP(sizeHidden = 4L, numLayers = 1L, sizeEmbedding = 4L)
  )
  for (settings in settingsList) {
    parameters <- settings$param[[1]]
    parameters$modelType <- settings$modelType
    parameters$feature_info <- dataset$get_feature_info()
    estimator <- createEstimator(list(
      modelParameters = parameters,
      estimatorSettings = settings$estimatorSettings
    ))
    predictions <- as.numeric(estimator$predict_proba(dataset))
    expect_length(predictions, 4)
    expect_true(all(is.finite(predictions)), info = settings$modelType)
  }
})

test_that("training keeps patients without covariates aligned with their labels", {
  skip_if_no_python()
  data <- createMockData(data.frame(
    rowId = c(1, 3), covariateId = c(101, 101), columnId = c(1L, 1L),
    covariateValue = c(1, 2)
  ))
  withr::defer(Andromeda::close(data))
  dataset <- createDataset(data, data.frame(rowId = 1:3, outcomeCount = c(1, 0, 1)))
  expect_equal(reticulate::py_len(dataset), 3L)
  expect_equal(as.numeric(dataset$data$feature_values$numpy()), c(1, 0, 2))
})

test_that("covariates outside of the cohort are rejected", {
  skip_if_no_python()
  data <- createMockData(data.frame(
    rowId = 5, covariateId = 101, columnId = 1L, covariateValue = 1
  ))
  withr::defer(Andromeda::close(data))
  expect_error(
    createDataset(data, data.frame(rowId = 1:2, outcomeCount = c(0, 1))),
    "only 2 patients"
  )
})

test_that("a mismatch between predictions and the cohort is an error", {
  estimator <- list(predict_proba = function(data) c(0.1, 0.9))
  expect_error(
    predictDeepEstimator(estimator, data = NULL, cohort = data.frame(rowId = 1:4)),
    "2 predictions for 4 patients"
  )
})

test_that("the tuning metric picks the best hyperparameters", {
  results <- list(
    list(gridPerformance = list(cvPerformance = 0.2), prediction = "a"),
    NULL,
    list(gridPerformance = list(cvPerformance = 0.9), prediction = "c")
  )
  expect_equal(getBestGridIndex(results, maximize = TRUE), 3)
  expect_equal(getBestGridIndex(results, maximize = FALSE), 1)

  cache <- trainingCache$new(withr::local_tempdir())
  trimmed <- cache$trimPerformance(results, maximize = FALSE)
  expect_equal(trimmed[[1]]$prediction, "a")
  expect_equal(trimmed[[3]]$prediction, list(NULL))
})

test_that("search settings from PatientLevelPrediction are reported as ignored", {
  expect_warning(
    ignored <- warnIgnoredSearchSettings(
      PatientLevelPrediction::createHyperparameterSettings(
        search = "random", sampleSize = 2, randomSeed = 1
      )
    ),
    "ignores `search`, `sampleSize`, `randomSeed`"
  )
  expect_equal(ignored, c("search", "sampleSize", "randomSeed"))
  metric <- PatientLevelPrediction::createTuningMetric(
    fun = function(prediction) 1, maximize = FALSE, name = "constant"
  )
  expect_no_warning(warnIgnoredSearchSettings(
    PatientLevelPrediction::createHyperparameterSettings(tuningMetric = metric)
  ))
  expect_no_warning(warnIgnoredSearchSettings(NULL))
})

test_that("computeTuningPerformance uses the requested metric", {
  metric <- PatientLevelPrediction::createTuningMetric(
    fun = function(prediction) mean(prediction$value),
    maximize = FALSE,
    name = "meanRisk"
  )
  prediction <- data.frame(
    rowId = 1:4, outcomeCount = c(0, 1, 0, 1),
    value = c(0.1, 0.2, 0.3, 0.4), index = c(1, 1, 2, 2)
  )
  performance <- computeTuningPerformance(prediction, list(sizeHidden = 4), metric)
  expect_equal(performance$cvPerformance, 0.25)
  expect_equal(performance$metric, "meanRisk")
  expect_true(all(performance$hyperSummary$metric == "meanRisk"))
  expect_equal(
    computeTuningPerformance(prediction, list(sizeHidden = 4))$metric,
    "computeAuc"
  )
})

test_that("gridCvDeep selects the hyperparameters with the requested metric", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  settings <- setMultiLayerPerceptron(
    numLayers = 1, sizeHidden = c(4, 8), dropout = 0, sizeEmbedding = 4,
    estimatorSettings = setEstimator(
      learningRate = 0.01, batchSize = 4, epochs = 1, device = "cpu", seed = 42
    ),
    hyperParamSearch = "grid"
  )
  metric <- PatientLevelPrediction::createTuningMetric(
    fun = function(prediction) stop("not used with the mocked cross validation"),
    maximize = FALSE,
    name = "lowerIsBetter"
  )
  performance <- c(0.9, 0.1)
  fit <- function(tuningMetric) {
    testthat::with_mocked_bindings(
      gridCvDeep(
        mappedData = data, labels = labels, modelSettings = settings,
        modelLocation = withr::local_tempdir(.local_envir = parent.frame(3)),
        analysisPath = withr::local_tempdir(.local_envir = parent.frame(3)),
        tuningMetric = tuningMetric
      ),
      doCrossValidation = function(dataset, labels, parameters, modelSettings, tuningMetric) {
        prediction <- labels
        prediction$value <- 0.5
        index <- match(parameters$sizeHidden, c(4, 8))
        list(
          prediction = prediction,
          param = c(parameters, list(learnSchedule = list(LRs = list(0.01)))),
          gridPerformance = list(
            cvPerformance = performance[index],
            hyperSummary = data.frame(metric = "x")
          )
        )
      }
    )
  }
  expect_equal(fit(metric)$finalParam$sizeHidden, 8)
  expect_equal(fit(NULL)$finalParam$sizeHidden, 4)
})

temporalData <- function(timeRef, timeIds) {
  createMockData(
    data.frame(
      rowId = 1, covariateId = 101, columnId = 1L, covariateValue = 1, timeId = timeIds
    ),
    isBinary = "Y",
    timeRef = timeRef
  )
}

timeTokenNames <- function(data) {
  dataset <- createDataset(
    data,
    data.frame(rowId = 1, outcomeCount = 1),
    temporalSettings = list(maxSequenceLength = "max", timeTokens = TRUE)
  )
  reference <- reticulate::py_to_r(dataset$get_feature_info()$data_reference)
  ids <- as.integer(dataset$data$feature_ids$numpy())
  reference$covariateName[match(ids[ids > 1], reference$columnId)]
}

test_that("time tokens use the units of the time reference", {
  skip_if_no_python()
  cases <- list(
    list(part = "day", interval = 1, ids = c(0, 30), expected = "M1"),
    list(part = "day", interval = 30, ids = c(0, 1), expected = "M1"),
    list(part = "day", interval = 30, ids = c(0, 3), expected = "M3"),
    list(part = "month", interval = 1, ids = c(0, 3), expected = "M3"),
    list(part = "MONTH", interval = 1, ids = c(0, 12), expected = "LT"),
    list(part = "week", interval = 1, ids = c(0, 1), expected = "W1")
  )
  for (case in cases) {
    data <- temporalData(
      data.frame(timePart = case$part, timeInterval = case$interval),
      case$ids
    )
    expect_equal(timeTokenNames(data), case$expected, info = paste(case$part, case$interval))
    Andromeda::close(data)
  }
})

test_that("time tokens use the start days of explicit time windows", {
  skip_if_no_python()
  # createTemporalCovariateSettings() gives a timeRef of windows with start and end days
  windows <- data.frame(
    timeId = c(0, 1, 2), startDay = c(-400, -60, -30), endDay = c(-61, -31, -1)
  )
  for (case in list(list(ids = c(1, 2), expected = "M1"), list(ids = c(0, 2), expected = "LT"))) {
    data <- temporalData(windows, case$ids)
    expect_equal(timeTokenNames(data), case$expected, info = paste(case$ids, collapse = " "))
    Andromeda::close(data)
  }
})

test_that("time tokens reject time references without a usable unit", {
  skip_if_no_python()
  for (timeRef in list(
    data.frame(timeId = 1:2),
    data.frame(timePart = "fortnight", timeInterval = 1),
    data.frame(timePart = "day", timeInterval = 0)
  )) {
    data <- temporalData(timeRef, c(0, 1))
    expect_error(timeTokenNames(data), "[Tt]ime tokens")
    Andromeda::close(data)
  }
})

test_that("RealMLP mean aggregation ignores padding", {
  skip_if_no_python()
  model <- importModule("RealMLP")$RealMLP(
    feature_info = mappedFeatureInfo(), size_embedding = 2L, size_hidden = 4L,
    num_layers = 1L, dropout = 0, paper_mode = FALSE
  )
  expect_equal(model$token_aggregation, "mean")
  tokens <- torch$tensor(array(c(1, 3, 0, 0, 1, 3, 0, 0), dim = c(1, 4, 2)), dtype = torch$float)
  ids <- torch$tensor(matrix(c(1L, 2L, 0L, 0L), nrow = 1), dtype = torch$long)
  expect_equal(asNumeric(model$`_aggregate_tokens`(tokens, ids)), c(2, 2))
})

test_that("wide coefficients keep column and covariate IDs apart", {
  skip_if_no_python()
  realMlp <- importModule("RealMLP")$RealMLP
  # column 1 has covariateId 3, so column ID 3 must not be read as covariateId 3
  featureInfo <- mappedFeatureInfo(covariateIds = c(3, 9, 7, 8))
  loadWide <- function(lines) {
    file <- withr::local_tempfile(fileext = ".csv", .local_envir = parent.frame(2))
    writeLines(lines, file)
    model <- realMlp(
      feature_info = featureInfo, size_embedding = 2L, size_hidden = 4L,
      num_layers = 1L, dropout = 0, paper_mode = FALSE,
      wide_enabled = TRUE, wide_init_path = file
    )
    as.numeric(model$wide_embedding$weight$detach()$numpy())[2:5]
  }
  expect_equal(loadWide(c("columnId,weight", "3,2")), c(0, 0, 2, 0))
  expect_equal(loadWide(c("covariateId,weight", "3,2")), c(2, 0, 0, 0))
  expect_equal(loadWide(c("covariateId,columnId,weight", "3,1,2")), c(2, 0, 0, 0))
  # not in the vocabulary, so it can not be used as a column ID either
  expect_error(loadWide(c("covariateId,weight", "2,2")), "Failed to map")
  # a file's columnId comes from the model that wrote it, so covariateId decides
  expect_equal(loadWide(c("covariateId,columnId,weight", "3,2,2")), c(2, 0, 0, 0))
  # a covariateImportance export from another model
  expect_equal(
    loadWide(c(
      "covariateId,covariateName,columnId,covariateValue",
      "9,second,5,1.5", "8,fourth,1,0.5", "99,unknown,2,4"
    )),
    c(0, 1.5, 0, 0.5)
  )
})

test_that("data-dependent initialization leaves frozen RealMLP weights alone", {
  skip_if_no_python()
  realMlp <- importModule("RealMLP")$RealMLP
  batch <- list(
    feature_ids = torch$tensor(matrix(c(1L, 2L, 3L, 4L), nrow = 2), dtype = torch$long),
    feature_values = torch$ones(2L, 2L, dtype = torch$float)
  )
  layerWeights <- function(freeze) {
    torch$manual_seed(42L)
    model <- realMlp(
      feature_info = mappedFeatureInfo(), size_embedding = 2L, size_hidden = 4L,
      num_layers = 1L, dropout = 0, paper_mode = FALSE,
      wide_enabled = TRUE, freeze_deep = freeze
    )
    layer <- model$`_linear_layers`[[1]]
    before <- c(asNumeric(layer$weight), asNumeric(layer$bias))
    model$data_dependent_init(list(batch))
    list(before = before, after = c(asNumeric(layer$weight), asNumeric(layer$bias)))
  }
  frozen <- layerWeights(TRUE)
  expect_equal(frozen$after, frozen$before)
  trainable <- layerWeights(FALSE)
  expect_false(isTRUE(all.equal(trainable$after, trainable$before)))
})

test_that("convolutional positional encodings do not depend on padding", {
  skip_if_no_python()
  encodings <- importModule("PositionalEncodings")
  head <- function(tensor, n) as.numeric(tensor$detach()$numpy()[1, , 1:n, ])
  attentionPe <- function(module, length) {
    q <- torch$zeros(1L, 2L, length, 2L)
    timeIds <- torch$arange(length + 0L, dtype = torch$long)$unsqueeze(0L)
    module$apply_attention_pe(q, q, time_ids = timeIds)
  }
  modules <- list(
    encodings$StochasticConvPE(dim_token = 4L, num_heads = 2L, kernel_size = 3L),
    encodings$HybridRoPEConvPE(dim_token = 4L, max_time_id = 16L, num_heads = 2L, kernel_size = 3L)
  )
  for (module in modules) {
    short <- attentionPe(module, 3L)
    long <- attentionPe(module, 5L)
    expect_equal(head(short[[1]], 3L), head(long[[1]], 3L), tolerance = 1e-6)
    expect_equal(head(short[[2]], 3L), head(long[[2]], 3L), tolerance = 1e-6)
  }
})

test_that("TemporalPE uses the requested sigma", {
  skip_if_no_python()
  temporalPE <- importModule("PositionalEncodings")$TemporalPE
  for (learnable in c(FALSE, TRUE)) {
    module <- temporalPE(
      dim_token = 4L, num_heads = 2L, sigma = 2, learnable_sigma = learnable
    )
    expect_equal(torch$exp(module$log_sigma)$item(), 2, tolerance = 1e-6)
  }
  expect_error(temporalPE(dim_token = 4L, num_heads = 2L, sigma = 0), "must be positive")
})

test_that("memory cleanup leaves tensors that are still in use intact", {
  skip_if_no_python()
  importModule("gpu_memory_cleanup")
  # Pretend that GPU memory use stays high after garbage collection and that a
  # tensor still in use is on the GPU, which is when cleanup used to free the
  # storage of every tensor it found.
  result <- reticulate::py_run_string("
def cleanup_with_high_memory_use():
    import warnings
    import torch
    import gpu_memory_cleanup as cleanup

    live = torch.arange(4.0)
    saved = {name: getattr(torch.cuda, name) for name in ('memory_allocated', 'empty_cache')}
    saved_get_tensors = getattr(cleanup, 'get_tensors', None)
    torch.cuda.memory_allocated = lambda *args, **kwargs: 4 * 1024 ** 3
    torch.cuda.empty_cache = lambda: None
    cleanup.get_tensors = lambda gpu_only=True: iter([live])
    try:
        # warnings from the cleanup code are beside the point here
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cleanup.memory_cleanup()
    except Exception:
        pass
    finally:
        for name, value in saved.items():
            setattr(torch.cuda, name, value)
        if saved_get_tensors is None:
            del cleanup.get_tensors
        else:
            cleanup.get_tensors = saved_get_tensors
    return live.untyped_storage().nbytes() // live.element_size()

remaining = cleanup_with_high_memory_use()
", local = TRUE)
  expect_equal(result$remaining, 4)
})

poincareEmbedding <- function(points, aggregate = "none", ...) {
  importModule("CustomEmbeddings")$PoincareEmbeddings(
    custom_embedding_weights = torch$tensor(points, dtype = torch$float),
    embedding_dim = ncol(points),
    feature_info = mappedFeatureInfo(),
    custom_indices = torch$tensor(array(seq_len(nrow(points))), dtype = torch$long),
    aggregate = aggregate,
    ...
  )
}

embedTokens <- function(embedding, ids) {
  embedding(list(
    feature_ids = torch$tensor(matrix(as.integer(ids), nrow = 1), dtype = torch$long),
    feature_values = torch$ones(c(1L, length(ids)), dtype = torch$float)
  ))
}

test_that("Poincare embeddings add the trainable offset in the tangent space", {
  skip_if_no_python()
  point <- c(0.3, 0.4)
  offset <- c(0.2, -0.1)
  embedding <- poincareEmbedding(rbind(point))
  embedding$custom_embeddings_trainable$weight$data$copy_(
    torch$tensor(rbind(c(0, 0), offset), dtype = torch$float)
  )
  # the log map at the origin of a point x with curvature 1
  tangent <- atanh(sqrt(sum(point^2))) / sqrt(sum(point^2)) * point
  tokens <- embedTokens(embedding, c(1, 2))
  expect_equal(asNumeric(tokens$select(1L, 0L)$reshape(-1L)), tangent + offset, tolerance = 1e-6)
})

test_that("Poincare embeddings must lie inside the ball of their curvature", {
  skip_if_no_python()
  # a norm of 0.6 is inside the ball with curvature -1 but not with -4
  point <- rbind(c(0.36, 0.48))
  expect_no_error(poincareEmbedding(point))
  expect_error(poincareEmbedding(point, curvature = 4), "not inside the Poincare ball")
  expect_error(poincareEmbedding(rbind(c(0.6, 0.8))), "not inside the Poincare ball")
  expect_error(poincareEmbedding(rbind(c(NaN, 0.1))), "not finite")
  expect_error(poincareEmbedding(point, curvature = 0), "positive number")
})

test_that("the Poincare log map uses the curvature and is finite at the origin", {
  skip_if_no_python()
  logmap0 <- importModule("CustomEmbeddings")$logmap0
  point <- c(0.3, 0.4)
  for (curvature in c(1, 2)) {
    radius <- sqrt(curvature) * sqrt(sum(point^2))
    mapped <- logmap0(torch$tensor(rbind(point), dtype = torch$float), curvature)
    expect_equal(asNumeric(mapped), atanh(radius) / radius * point, tolerance = 1e-6)
  }
  origin <- torch$zeros(c(1L, 2L), dtype = torch$float, requires_grad = TRUE)
  mapped <- logmap0(origin, 1)
  mapped$sum()$backward()
  expect_equal(asNumeric(mapped), c(0, 0))
  expect_equal(asNumeric(origin$grad), c(1, 1))
})

test_that("setCustomEmbeddingModel passes the curvature to Poincare embeddings", {
  skip_if_no_python()
  embeddingFile <- withr::local_tempfile(fileext = ".pt")
  torch$save(list(
    concept_ids = torch$tensor(array(1001L), dtype = torch$long),
    embeddings = torch$tensor(rbind(c(0.3, 0.4)), dtype = torch$float)
  ), embeddingFile)
  settings <- setCustomEmbeddingModel(
    embeddingFilePath = embeddingFile,
    modelSettings = setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 4, dropout = 0, sizeEmbedding = 2,
      estimatorSettings = setEstimator(learningRate = 0.001, device = "cpu", seed = 42),
      randomSample = 1
    ),
    embeddingsClass = "PoincareEmbeddings",
    curvature = 2
  )
  parameters <- settings$param[[1]]
  parameters$modelType <- settings$modelType
  parameters$feature_info <- mappedFeatureInfo()
  estimator <- createEstimator(list(
    modelParameters = parameters,
    estimatorSettings = settings$estimatorSettings
  ))
  expect_equal(estimator$model$embedding$curvature, 2)
  expect_error(
    setCustomEmbeddingModel(embeddingFilePath = embeddingFile, curvature = 2),
    "only applies to"
  )
})

test_that("the projection of custom embeddings only applies to custom features", {
  skip_if_no_python()
  # custom embeddings with two dimensions projected to three
  customEmbedding <- function(aggregate) {
    importModule("CustomEmbeddings")$CustomEmbeddings(
      custom_embedding_weights = torch$tensor(rbind(c(1, 2)), dtype = torch$float),
      embedding_dim = 3L,
      feature_info = mappedFeatureInfo(),
      custom_indices = torch$tensor(array(1L), dtype = torch$long),
      aggregate = aggregate
    )
  }
  position <- function(tokens, i) asNumeric(tokens$select(1L, i)$reshape(-1L))
  torch$manual_seed(42L)

  perToken <- customEmbedding("none")
  projected <- asNumeric(perToken$linear_transform(torch$tensor(rbind(c(1, 2)), dtype = torch$float)))
  regular <- embedTokens(perToken$regular_embeddings, c(0, 2, 0))
  tokens <- embedTokens(perToken, c(1, 2, 0))
  expect_equal(position(tokens, 0L), projected, tolerance = 1e-6)
  expect_equal(position(tokens, 1L), position(regular, 1L), tolerance = 1e-6)
  expect_equal(position(tokens, 2L), c(0, 0, 0))

  pooled <- customEmbedding("sum")
  regularOnly <- asNumeric(embedTokens(pooled$regular_embeddings, c(2, 0)))
  expect_equal(asNumeric(embedTokens(pooled, c(2, 0))), regularOnly, tolerance = 1e-6)
  projected <- asNumeric(pooled$linear_transform(torch$tensor(rbind(c(1, 2)), dtype = torch$float)))
  expect_equal(
    asNumeric(embedTokens(pooled, c(1, 2))), (projected + regularOnly) / 2, tolerance = 1e-6
  )
})

test_that("the search frees a combination that runs out of memory and continues", {
  skip_if_no_python()
  reticulate::py_run_string("
import gc
import weakref
import torch

failed_models = []

def run_out_of_memory():
    model = torch.nn.Linear(2, 1)  # stands in for a model on the GPU
    failed_models.append(weakref.ref(model))
    raise torch.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB')

def failed_model_freed():
    gc.collect()
    return failed_models[-1]() is None
")
  main <- reticulate::import_main()
  labels <- createMockLabels()
  result <- testthat::with_mocked_bindings(
    doCrossValidation(
      dataset = NULL, labels = labels, parameters = list(numLayers = 1),
      modelSettings = list()
    ),
    doCrossValidationImpl = function(...) main$run_out_of_memory()
  )
  expect_true(all(is.na(result$prediction$value)))
  expect_equal(nrow(result$prediction), nrow(labels))
  expect_true(main$failed_model_freed())
})
