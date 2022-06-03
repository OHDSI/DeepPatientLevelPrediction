context("TabNet")

tabSet <- setTabNetTorch(
  batch_size = 256,
  penalty = 1e-3,
  clip_value = NULL,
  loss = "auto",
  epochs = 5,
  drop_last = FALSE,
  decision_width = 8,
  attention_width = 8,
  num_steps = 3,
  feature_reusage = 1.3,
  mask_type = "sparsemax",
  virtual_batch_size = 128,
  valid_split = 0,
  learn_rate = 2e-2,
  optimizer = "adam",
  lr_scheduler = NULL,
  lr_decay = 0.1,
  step_size = 30,
  checkpoint_epochs = 10,
  cat_emb_dim = 1,
  num_independent = 2,
  num_shared = 2,
  momentum = 0.02,
  pretraining_ratio = 0.5,
  verbose = FALSE,
  device = "auto",
  importance_sample_size = 1e5,
  seed=NULL,
  hyperParamSearch = 'random',
  randomSample = 100
  )

test_that("setTabNet works", {
  
  testthat::expect_s3_class(object = tabSet, class = 'modelSettings')
  
  testthat::expect_equal(tabSet$fitFunction, 'fitTabNetTorch') 
  
  testthat::expect_true(length(tabSet$param) > 0 )
  
})


tab2 <- res <- tryCatch({
  PatientLevelPrediction::runPlp(
    plpData = plpData, 
    outcomeId = 3, 
    modelSettings = tabSet,
    analysisId = 'TabNet', 
    analysisName = 'Testing TabNet', 
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

test_that("setTabNetTorch with runPlp working checks", {
  
  testthat::expect_false(is.null(tab2))
  
  # check structure
  testthat::expect_true('prediction' %in% names(tab2))
  testthat::expect_true('model' %in% names(tab2))
  testthat::expect_true('covariateSummary' %in% names(tab2))
  testthat::expect_true('performanceEvaluation' %in% names(tab2))
  
  # check prediction same size as pop
  testthat::expect_equal(nrow(tab2$prediction), nrow(population))
  
  # check prediction between 0 and 1
  testthat::expect_gte(min(tab2$prediction$value), 0)
  testthat::expect_lte(max(tab2$prediction$value), 1)
  
})
