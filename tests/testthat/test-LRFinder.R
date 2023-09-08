ResNet <- reticulate::import_from_path("ResNet", path)$ResNet
lrFinderClass <- reticulate::import_from_path("LrFinder", path=path)$LrFinder

test_that("LR scheduler that changes per batch works", {
  
  model <- ResNet(cat_features = 10L, num_features = 1L,
                  size_embedding = 32L, size_hidden = 64L,
                  num_layers = 1L, hidden_factor = 1L)
  optimizer <- torch$optim$AdamW(model$parameters(), lr=1e-7)
  
  ExponentialSchedulerPerBatch <- reticulate::import_from_path("LrFinder", 
                                                               path=path)$ExponentialSchedulerPerBatch
  scheduler <- ExponentialSchedulerPerBatch(optimizer,
                          end_lr = 1e-2,
                          num_iter = 5)
  expect_equal(scheduler$last_epoch, 0)
  expect_equal(scheduler$optimizer$param_groups[[1]]$lr, 1e-7)
  
  for (i in 1:5) {
    optimizer$step()
    scheduler$step()
  }
  
  expect_equal(scheduler$last_epoch, 5)
  expect_equal(scheduler$optimizer$param_groups[[1]]$lr, (1e-7 * (0.01 / 1e-7) ^ (5 / 4)))
  
})


test_that("LR finder works", {
  lrFinder <- createLRFinder(modelType="ResNet",
                             modelParameters = list(cat_features=length(dataset$get_cat_features()),
                                                    num_features=length(dataset$get_numerical_features()),
                                                    size_embedding=32L,
                                                    size_hidden=64L,
                                                    num_layers=1L,
                                                    hidden_factor=1L),
                             estimatorSettings = setEstimator(batchSize = 32L,
                                                              seed=42),
                             lrSettings = list(minLr=3e-4,
                                               maxLr=10.0,
                                               numLr=20L,
                                               divergenceThreshold=1.1))
    
  lr <- lrFinder$get_lr(dataset)
  
  expect_true(lr<=10.0)
  expect_true(lr>=3e-4)
})

test_that("LR finder works with device specified by a function", {
  
  deviceFun <- function(){
    dev = "cpu"
  }
  lrFinder <- createLRFinder(model = "ResNet", 
                             modelParameters = list(cat_features=length(dataset$get_cat_features()),
                                                num_features=length(dataset$get_numerical_features()),
                                                size_embedding=8L,
                                                size_hidden=16L,
                                                num_layers=1L,
                                                hidden_factor=1L),
                             estimatorSettings = setEstimator(batchSize=32L,
                                                  seed = 42,
                                                  device = deviceFun),
                             lrSettings = list(minLr=3e-4,
                                               maxLr=10.0,
                                               numLr=20L,
                                               divergenceThreshold=1.1)
                             )
  lr <- lrFinder$get_lr(dataset)
  
  expect_true(lr<=10.0)
  expect_true(lr>=3e-4)
  
  
})