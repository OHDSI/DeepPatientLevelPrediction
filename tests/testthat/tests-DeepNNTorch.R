# code to train models
deepset <- setDeepNNTorch(units=list(c(128, 64), 128), layer_dropout=c(0.2),
                          lr =c(1e-4), decay=c(1e-5), outcome_weight = c(1.0), batch_size = c(100), 
                          epochs= c(5),  seed=NULL  )

test_that("setDeepNNTorch works", {
  
  testthat::expect_s3_class(object = deepset, class = 'modelSettings')
  
  testthat::expect_equal(deepset$fitFunction, 'fitDeepNNTorch') 
  
  testthat::expect_true(nrow(deepset$param) > 0 )
  
})

sink(nullfile())
res <- tryCatch({
  PatientLevelPrediction::runPlp(
    plpData = plpData, 
    outcomeId = 3, 
    modelSettings = deepset,
    analysisId = 'DeepNNTorch', 
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
    saveDirectory = file.path(testLoc, 'DeepNNTorch')
  )
}, error = function(e){print(e); return(NULL)}
)
sink()

test_that("setDeepNNTorch with runPlp working checks", {
  
  testthat::expect_false(is.null(res))
  
  # check structure
  testthat::expect_true('prediction' %in% names(res))
  testthat::expect_true('model' %in% names(res))
  testthat::expect_true('covariateSummary' %in% names(res))
  testthat::expect_true('performanceEvaluation' %in% names(res))
  
  # check prediction same size as pop
  testthat::expect_equal(nrow(res$prediction %>% filter(evaluationType %in% c('Train', 'Test'))), 
                         nrow(population))
  
  # check prediction between 0 and 1
  testthat::expect_gte(min(res$prediction$value), 0)
  testthat::expect_lte(max(res$prediction$value), 1)
  
})

test_that("Triple layer-nn works", {
  deepset <- setDeepNNTorch(units=list(c(64,64,32), c(64,32,16), c(32,16,8)), layer_dropout=c(0.2),
                            lr =c(1e-4), decay=c(1e-5), outcome_weight = c(1.0), batch_size = c(100), 
                            epochs= c(5),  seed=NULL)
  
  sink(nullfile())
  results <- fitDeepNNTorch(trainData$Train, deepset$param, analysisId=1)
  sink()
  
  expect_equal(class(results), 'plpModel')
  expect_equal(attr(results, 'modelType'), 'binary')
  expect_equal(attr(results, 'saveType'), 'file')
  
  # check prediction between 0 and 1
  testthat::expect_gt(min(results$prediction$value), 0)
  testthat::expect_lt(max(results$prediction$value), 1)
})