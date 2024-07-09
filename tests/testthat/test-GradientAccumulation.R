

generateData <- function(observations, features, totalFeatures = 6,
                         numCovs = FALSE) {
  rowId <- rep(1:observations, each = features)
  withr::with_seed(42, {
    columnId <- sample(1:totalFeatures, observations * features, replace = TRUE)
  })
  covariateValue <- rep(1, observations * features)
  covariates <- data.frame(rowId = rowId, columnId = columnId, covariateValue = covariateValue)  
  if (numCovs) {
    numRow <- 1:observations
    numCol <- rep(totalFeatures + 1, observations)
    withr::with_seed(42, {
      numVal <- runif(observations)
    })
    numCovariates <- data.frame(rowId = as.integer(numRow),
                                columnId = as.integer(numCol),
                                covariateValue = numVal)
    covariates <- rbind(covariates, numCovariates)
  }
  
  labels <- as.numeric(sample(0:1, observations, replace = TRUE))
  data <- Andromeda::andromeda(covariates = covariates)
  data <- attr(data, "dbname")
  Data <- reticulate::import_from_path("Dataset", path = path)$Data
  dataset <- Data(data, labels)
  return(dataset)
}

dataset <- generateData(observations = 5, features = 3, totalFeatures = 6)

model <- reticulate::import_from_path("ResNet", path = path)$ResNet
modelParams <- list("num_layers" = 1L,
                    "size_hidden" = 32,
                    "size_embedding" = 32,
                    "cat_features" = dataset$get_cat_features()$max(),
                    "normalization" = torch$nn$LayerNorm)
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

estimator <- Estimator(
  model = model,
  model_parameters = modelParams,
  estimator_settings = estimatorSettings
)


test_that("Gradient accumulation provides same results as without, ", {
  predsWithout <- estimator$predict_proba(dataset)
  
  estimatorSettings$accumulation_steps <- 2
  estimatorWith <- Estimator(
    model = model,
    model_parameters = modelParams,
    estimator_settings = estimatorSettings
  )
  predsWith <- estimatorWith$predict_proba(dataset)
  
  expect_equal(predsWithout, predsWith, tolerance = 1e-4)
  
  dataset <- generateData(observations = 5, features = 3, totalFeatures = 6, 
                          numCovs = TRUE) 
  modelParams$num_features <- 1L 
  estimatorWith <- Estimator(
    model = model,
    model_parameters = modelParams,
    estimator_settings = estimatorSettings
  )
  estimatorSettings$accumulation_steps <- 1
  estimator <- Estimator(
    model = model,
    model_parameters = modelParams,
    estimator_settings = estimatorSettings
  )
  predsWith <- estimator$predict_proba(dataset)
  predsWithout <- estimatorWith$predict_proba(dataset)
  
  expect_equal(predsWithout, predsWith, tolerance = 1e-4)
})

test_that("Loss is equal without dropout and layernorm",  {
  dataloader <- torch$utils$data$DataLoader(dataset, batch_size = NULL,
                                            sampler = torch$utils$data$BatchSampler(
                                              torch$utils$data$RandomSampler(dataset),
                                              batch_size = 4L,
                                              drop_last = FALSE
                                            ))  
  loss <- estimator$fit_epoch(dataloader)
  
  estimatorSettings$accumulation_steps <- 2
  estimatorWith <- Estimator(
    model = model,
    model_parameters = modelParams,
    estimator_settings = estimatorSettings
  )
  lossWith <- estimatorWith$fit_epoch(dataloader)
  
  expect_equal(loss, lossWith, tolerance = 1e-2)
  
})

test_that("LR finder works same with and without accumulation", {
  modelParams$modelType <- "ResNet"
  
  lrSettings <- list(minLr = 1e-8,
                     maxLr = 0.01,
                     numLr = 10L,
                     divergenceThreshold = 1.1)

  lr <- getLR(estimatorSettings = estimatorSettings,
              modelParameters = modelParams,
              dataset = dataset,
              lrSettings = lrSettings)
  
  estimatorSettings$accumulation_steps <- 2
 
  lrWith <- getLR(estimatorSettings = estimatorSettings,
                  modelParameters = modelParams,
                  dataset = dataset,
                  lrSettings = lrSettings) 
  
  expect_equal(lr, lrWith, tolerance = 1e-10)
})

test_that("Gradient accumulation works when batch is smaller than batch_size * steps", {
  dataset <- generateData(observations = 50, features = 3, totalFeatures = 6)
  
  model <- reticulate::import_from_path("ResNet", path = path)$ResNet
  modelParams <- list("num_layers" = 1L,
                      "size_hidden" = 32,
                      "size_embedding" = 32,
                      "cat_features" = 6,
                      "normalization" = torch$nn$LayerNorm)
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
  
  estimator <- Estimator(
    model = model,
    model_parameters = modelParams,
    estimator_settings = estimatorSettings
  ) 
  
  preds <- estimator$predict_proba(dataset)
  
  expect_equal(length(preds), 50)
  expect_true(all(preds >= 0))
  expect_true(all(preds <= 1))
})
