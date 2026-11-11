skip_if_no_python()

# A dataset of random binary features, plus one numeric feature if numCovs.
generateData <- function(observations, features, totalFeatures = 6,
                         numCovs = FALSE) {
  withr::with_seed(42, {
    covariates <- data.frame(
      rowId = rep(1:observations, each = features),
      columnId = sample(1:totalFeatures, observations * features, replace = TRUE),
      covariateValue = 1
    )
    numericColumns <- NULL
    if (numCovs) {
      numericColumns <- as.integer(totalFeatures + 1)
      covariates <- rbind(covariates, data.frame(
        rowId = 1:observations,
        columnId = numericColumns,
        covariateValue = runif(observations)
      ))
    }
    labels <- list(outcomeCount = as.numeric(sample(0:1, observations, replace = TRUE)))
  })
  data <- createMockData(covariates, isBinary = "Y", numericColumns = numericColumns)
  createDataset(data, labels)
}

dataset <- generateData(observations = 5, features = 3, totalFeatures = 6)

model <- importModule("ResNet")$ResNet
modelParams <- list(
  "num_layers" = 1L,
  "size_hidden" = 32,
  "size_embedding" = 32,
  "feature_info" = dataset$get_feature_info(),
  "normalization" = torch$nn$LayerNorm
)
Estimator <- importModule("Estimator")$Estimator
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
# a new estimator for each test, so that training in one does not affect another
newEstimator <- function() Estimator(model = model, parameters = parameters)


test_that("Gradient accumulation provides same results as without, ", {
  estimator <- newEstimator()
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
  estimator <- newEstimator()
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

  model <- importModule("ResNet")$ResNet
  modelParams <- list(
    "num_layers" = 1L,
    "size_hidden" = 32,
    "size_embedding" = 32,
    "feature_info" = dataset$get_feature_info(),
    "normalization" = torch$nn$LayerNorm
  )
  Estimator <- importModule("Estimator")$Estimator
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
