
covariates <- data.frame(rowId = as.integer(c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4)),
                         columnId = as.integer(c(1, 4, 5, 2, 4, 0, 3, 1, 0, 4, 2, 5)),
                         covariateValue = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
labels <- c(1, 0, 1, 0)
data <- Andromeda::andromeda(covariates = covariates)
data <- attr(data, "dbname")
Data <- reticulate::import_from_path("Dataset", path = path)$Data
dataset <- Data(data, labels) 
  
model <- reticulate::import_from_path("ResNet", path = path)$ResNet
modelParams <- list("num_layers" = 1L,
                    "size_hidden" = 32,
                    "size_embedding" = 32,
                    "cat_features" = 5,
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
  
  expect_equal(predsWithout, predsWith, tolerance = 1e-5)
  
  numCovariates <- data.frame(rowId = as.integer(c(1, 2, 3, 4)),
                               columnId = as.integer(c(6, 6, 6, 6)),
                               covariateValue = c(0.5, 0.2, 0.3, 0.1))
  allCovariates <- rbind(covariates, numCovariates)
  data <- Andromeda::andromeda(covariates = allCovariates)
  data <- attr(data, "dbname")
  dataset <- Data(data, labels) 
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
  
  expect_equal(predsWithout, predsWith, tolerance = 1e-5)
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
  
  expect_equal(loss, lossWith)
  
})
