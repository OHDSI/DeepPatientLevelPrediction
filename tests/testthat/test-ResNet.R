context("ResNet")

resSet <- setResNet(
  numLayers = list(5), 
  sizeHidden = list(256),
  hiddenFactor = list(2),
  residualDropout = list(0.1), 
  hiddenDropout = list(0.1),
  normalization = list('BatchNorm'), 
  activation = list('RelU'),
  sizeEmbedding = list(64), 
  weightDecay = list(1e-6),
  learningRate = list(3e-4), 
  seed = 42, 
  hyperParamSearch = 'random',
  randomSample = 1, 
  #device='cuda:0', 
  batchSize = 128, 
  epochs = 10
)

test_that("setResNet works", {
  
  testthat::expect_s3_class(object = resSet, class = 'modelSettings')
  
  testthat::expect_equal(resSet$fitFunction, 'fitEstimator') 
  
  testthat::expect_true(length(resSet$param) > 0 )
  
})


res2 <- res <- tryCatch({
  PatientLevelPrediction::runPlp(
    plpData = plpData, 
    outcomeId = 3, 
    modelSettings = resSet,
    analysisId = 'ResNet', 
    analysisName = 'Testing Deep Learning', 
    populationSettings = populationSet, 
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(), 
    sampleSettings = PatientLevelPrediction::createSampleSettings(),  # none 
    featureEngineeringSettings = PatientLevelPrediction::createFeatureEngineeringSettings(), # none 
    preprocessSettings = PatientLevelPrediction::createPreprocessSettings(), 
    executeSettings = PatientLevelPrediction::createExecuteSettings(
      runSplitData = T, 
      runSampleData = F, 
      runfeatureEngineering = F, 
      runPreprocessData = T, 
      runModelDevelopment = T, 
      runCovariateSummary = F
    ), 
    saveDirectory = file.path(testLoc, 'Deep')
  )
}, error = function(e){print(e); return(NULL)}
)

test_that("setDeepNNTorch with runPlp working checks", {
  
  testthat::expect_false(is.null(res2))
  
  # check structure
  testthat::expect_true('prediction' %in% names(res2))
  testthat::expect_true('model' %in% names(res2))
  testthat::expect_true('covariateSummary' %in% names(res2))
  testthat::expect_true('performanceEvaluation' %in% names(res2))
  
  # check prediction same size as pop
  testthat::expect_equal(nrow(res2$prediction), nrow(population))
  
  # check prediction between 0 and 1
  testthat::expect_gte(min(res2$prediction$value), 0)
  testthat::expect_lte(max(res2$prediction$value), 1)
  
})
