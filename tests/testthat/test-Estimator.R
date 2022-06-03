test_that('Estimator fit function works', {
  
  # create a simple model
  modelSettings <- setResNet(numLayers=1, sizeHidden=16, hiddenFactor=1,
                             residualDropout=0, hiddenDropout=0, 
                             sizeEmbedding = 16, hyperParamSearch = 'random',
                             randomSample = 1, epochs=1)
  sink(nullfile())  
  results <- fitEstimator(trainData, param = modelSettings$param, analysisId = 1)
  sink()
  expect_true(!is.null(results$trainDetails$trainingTime))
  
  
})