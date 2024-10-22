resNet <- reticulate::import_from_path("ResNet", path)$ResNet

test_that("LR scheduler that changes per batch works", {

  model <- resNet(feature_info = list(categorical_features = 10L,
                                      numerical_features = 1L),
                  size_embedding = 32L, size_hidden = 64L,
                  num_layers = 1L, hidden_factor = 1L)
  optimizer <- torch$optim$AdamW(model$parameters(), lr = 1e-7)


  exponentialSchedulerPerBatch <-
    reticulate::import_from_path("LrFinder",
                                 path = path)$ExponentialSchedulerPerBatch
  scheduler <- exponentialSchedulerPerBatch(optimizer,
                                            end_lr = 1e-2,
                                            num_iter = 5)
  expect_equal(scheduler$last_epoch, 0)
  expect_equal(scheduler$optimizer$param_groups[[1]]$lr, 1e-7)

  for (i in 1:5) {
    optimizer$step()
    scheduler$step()
  }

  expect_equal(scheduler$last_epoch, 5)
  expect_equal(scheduler$optimizer$param_groups[[1]]$lr,
               (1e-7 * (0.01 / 1e-7) ^ (5 / 4)))

})


test_that("LR finder works", {
  estimatorSettings <- setEstimator(batchSize = 32L,
                                    seed = 42)
  modelParameters <- list(feature_info = dataset$get_feature_info(),
                          size_embedding = 32L,
                          size_hidden = 64L,
                          num_layers = 1L,
                          hidden_factor = 1L,
                          modelType = "ResNet")
 estimatorSettings <- estimatorSettings
 lrSettings <- list(minLr = 1e-8,
                    maxLr = 0.01,
                    numLr = 20L,
                    divergenceThreshold = 1.1)  
 # initial LR should be the minLR
  parameters <- list(
    modelParameters = modelParameters,
    estimatorSettings = estimatorSettings
  )
  lr <- getLR(parameters = parameters,
              lrSettings = lrSettings,
              dataset = dataset)
  tol <- 1e-10
  expect_lte(lr, 0.01 + tol)
  expect_gte(lr, 1e-08 - tol)
})

test_that("LR finder works with device specified by a function", {

  deviceFun <- function() {
    dev <- "cpu"
    dev
  }
  modelParameters <-
      list(feature_info = dataset$get_feature_info(),
           size_embedding = 8L,
           size_hidden = 16L,
           num_layers = 1L,
           hidden_factor = 1L,
           modelType = "ResNet")
  estimatorSettings <- setEstimator(batchSize = 32L,
                                    seed = 42,
                                    device = deviceFun)
  lrSettings <- list(minLr = 1e-6,
                     maxLr = 0.03,
                     numLr = 20L,
                     divergenceThreshold = 1.1)
  parameters <- list(modelParameters = modelParameters,
                     estimatorSettings = estimatorSettings)
  lr <- getLR(parameters = parameters,
              lrSettings = lrSettings,
              dataset = dataset)

  tol <- 1e-8
  expect_lte(lr, 0.03 + tol) 
  expect_gte(lr, 1e-6 - tol)
})
