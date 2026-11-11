# Shared helpers for tests: small synthetic data and settings that need no
# database, and access to the package's Python modules. testthat sources
# helper files before the tests, so these must not start Python when sourced.

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

mappedFeatureInfo <- function(covariateIds = 1:4 + 100, conceptIds = 1:4 + 1000,
                              isBinary = rep("Y", length(covariateIds))) {
  polars <- reticulate::import("polars", convert = FALSE)
  columnIds <- seq_along(covariateIds)
  importModule("Dataset")$FeatureInfo(polars$DataFrame(list(
    columnId = columnIds,
    covariateId = covariateIds,
    conceptId = conceptIds,
    isBinary = isBinary
  )))
}

# Evaluates an expression without its printed output and messages, so test
# logs stay readable. The output is restored even when the expression fails.
quietly <- function(expr) {
  withr::local_output_sink(nullfile())
  suppressMessages(expr)
}
