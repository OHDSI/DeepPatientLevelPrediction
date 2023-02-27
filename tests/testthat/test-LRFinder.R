test_that("LR scheduler that changes per batch works", {
  
  model <- ResNet(catFeatures = 10, numFeatures = 1,
                  sizeEmbedding = 32, sizeHidden = 64,
                  numLayers = 1, hiddenFactor = 1)
  optimizer <- torchopt::optim_adamw(model$parameters, lr=1e-7)
  
  scheduler <- lrPerBatch(optimizer,
                          startLR = 1e-7,
                          endLR = 1e-2,
                          nIters = 5)
  expect_equal(scheduler$last_epoch, 0)
  expect_equal(scheduler$optimizer$param_groups[[1]]$lr, 1e-7)
  
  for (i in 1:5) {
    scheduler$step()
  }
  
  expect_equal(scheduler$last_epoch, 5)
  expect_equal(scheduler$optimizer$param_groups[[1]]$lr, (1e-7 * (0.01 / 1e-7) ^ (5 / 4)))
  
})


test_that("LR finder works", {
  
  lr <- lrFinder(dataset, modelType = ResNet, modelParams = list(catFeatures=dataset$numCatFeatures(),
                                                           numFeatures=dataset$numNumFeatures(),
                                                           sizeEmbedding=32,
                                                           sizeHidden=64,
                                                           numLayers=1,
                                                           hiddenFactor=1),
           estimatorSettings = setEstimator(batchSize=32,
                                            seed = 42),
           minLR = 3e-4,
           maxLR = 10.0,
           numLR = 20,
           divergenceThreshold = 1.1)
  
  expect_true(lr<=10.0)
  expect_true(lr>=3e-4)
  
  
})