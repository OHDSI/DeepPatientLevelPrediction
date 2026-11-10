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
