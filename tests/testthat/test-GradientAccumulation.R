generateData <- function(observations, features, totalFeatures = 6,
                         numCovs = FALSE) {
  rowId <- rep(1:observations, each = features)
  withr::with_seed(42, {
    columnId <- sample(1:totalFeatures, observations * features, replace = TRUE)
  })
  covariateValue <- rep(1, observations * features)
  covariates <- data.frame(rowId = rowId, columnId = columnId, covariateValue = covariateValue)
  covariateRef <- data.frame(
    columnId = unique(covariates %>% dplyr::pull(.data$columnId)),
    analysisId = 1L
  )
  analysisRef <- data.frame(
    analysisId = 1L,
    analysisName = "test",
    isBinary = "Y",
    missingMeansZero = NA
  )
  if (numCovs) {
    numRow <- 1:observations
    numCol <- as.integer(rep(totalFeatures + 1, observations))
    withr::with_seed(42, {
      numVal <- runif(observations)
    })
    numCovariates <- data.frame(
      rowId = as.integer(numRow),
      columnId = as.integer(numCol),
      covariateValue = numVal
    )
    covariates <- rbind(covariates, numCovariates)
    covariateRef <- rbind(covariateRef, data.frame(
      columnId = unique(numCol),
      analysisId = 2L
    ))
    analysisRef <- rbind(analysisRef, data.frame(
      analysisId = 2L,
      analysisName = "numCov",
      isBinary = "N",
      missingMeansZero = "Y"
    ))
  }

  labels <- as.numeric(sample(0:1, observations, replace = TRUE))
  data <- Andromeda::andromeda(
    covariates = covariates,
    covariateRef = covariateRef,
    analysisRef = analysisRef)
  dataPath <- attr(data, "dbname")
  Data <- reticulate::import_from_path("Dataset", path = path)$Data
  dataset <- Data(dataPath, labels)
  return(dataset)
}

dataset <- generateData(observations = 5, features = 3, totalFeatures = 6)

model <- reticulate::import_from_path("ResNet", path = path)$ResNet
modelParams <- list(
  "num_layers" = 1L,
  "size_hidden" = 32,
  "size_embedding" = 32,
  "feature_info" = dataset$get_feature_info(),
  "normalization" = torch$nn$LayerNorm
)
Estimator <- reticulate::import_from_path("Estimator", path = path)$Estimator
estimatorSettings <- list(
  learningRate = 3e-4,
  weightDecay = 1e-6,
  device = "cpu",
  batch_size = 4,
  epochs = 1,
  seed = 32,
  find_l_r = FALSE,
  accumulation_steps = NULL,
  optimizer = torch$optim$AdamW,
  criterion = torch$nn$BCEWithLogitsLoss
)
parameters <- list(
  model_parameters = modelParams,
  estimator_settings = estimatorSettings
)
estimator <- Estimator(
  model = model,
  parameters = parameters
)


test_that("Gradient accumulation provides same results as without, ", {
  predsWithout <- estimator$predict_proba(dataset)

  estimatorSettings$accumulation_steps <- 2
  parameters <- list(
    model_parameters = modelParams,
    estimator_settings = camelCaseToSnakeCaseNames(estimatorSettings)
  )
  estimatorWith <- Estimator(
    model = model,
    parameters = parameters
  )
  predsWith <- estimatorWith$predict_proba(dataset)

  expect_equal(predsWithout, predsWith, tolerance = 1e-4)

  dataset <- generateData(
    observations = 5, features = 3, totalFeatures = 6,
    numCovs = TRUE
  )
  parameters$model_parameters$feature_info <- dataset$get_feature_info()

  estimatorWith <- Estimator(
    model = model,
    parameters = parameters
  )
  parameters$estimator_settings$accumulation_steps <- 1
  estimator <- Estimator(
    model = model,
    parameters = parameters
  )
  predsWith <- estimator$predict_proba(dataset)
  predsWithout <- estimatorWith$predict_proba(dataset)

  expect_equal(predsWithout, predsWith, tolerance = 1e-4)
})

test_that("Loss is equal without dropout and layernorm", {
  dataloader <- torch$utils$data$DataLoader(dataset,
    batch_size = NULL,
    sampler = torch$utils$data$BatchSampler(
      torch$utils$data$RandomSampler(dataset),
      batch_size = 4L,
      drop_last = FALSE
    )
  )
  loss <- estimator$fit_epoch(dataloader)

  parameters$estimator_settings$accumulation_steps <- 2
  estimatorWith <- Estimator(
    model = model,
    parameters = parameters
  )
  lossWith <- estimatorWith$fit_epoch(dataloader)

  expect_equal(loss, lossWith, tolerance = 1e-2)
})

test_that("LR finder works same with and without accumulation", {
  parameters$modelParameters <- modelParams
  parameters$modelParameters$modelType <- "ResNet"
  parameters$estimatorSettings <- estimatorSettings
  lrSettings <- list(
    minLr = 1e-8,
    maxLr = 0.01,
    numLr = 10L,
    divergenceThreshold = 1.1
  )

  lr <- getLR(
    parameters = parameters,
    dataset = dataset,
    lrSettings = lrSettings
  )

  parameters$estimatorSettings$accumulation_steps <- 2

  lrWith <- getLR(
    parameters = parameters,
    dataset = dataset,
    lrSettings = lrSettings
  )

  expect_equal(lr, lrWith, tolerance = 1e-10)
})

test_that("Gradient accumulation works when batch is smaller than batch_size * steps", {
  dataset <- generateData(observations = 50, features = 3, totalFeatures = 6)

  model <- reticulate::import_from_path("ResNet", path = path)$ResNet
  modelParams <- list(
    "num_layers" = 1L,
    "size_hidden" = 32,
    "size_embedding" = 32,
    "feature_info" = dataset$get_feature_info(),
    "normalization" = torch$nn$LayerNorm
  )
  Estimator <- reticulate::import_from_path("Estimator", path = path)$Estimator
  estimatorSettings <- list(
    learningRate = 3e-4,
    weightDecay = 1e-6,
    device = "cpu",
    batch_size = 20,
    epochs = 1,
    seed = 32,
    find_l_r = FALSE,
    accumulation_steps = 4,
    optimizer = torch$optim$AdamW,
    criterion = torch$nn$BCEWithLogitsLoss
  )
  parameters <- list(
    model_parameters = camelCaseToSnakeCaseNames(modelParams),
    estimator_settings = camelCaseToSnakeCaseNames(estimatorSettings)
  )
  estimator <- Estimator(
    model = model,
    parameters = parameters
  )

  preds <- estimator$predict_proba(dataset)

  expect_equal(length(preds), 50)
  expect_true(all(preds >= 0))
  expect_true(all(preds <= 1))
})
