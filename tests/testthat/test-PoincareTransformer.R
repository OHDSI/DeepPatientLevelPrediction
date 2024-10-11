
test_that("Poincare Transformer works", {
  settings <- setCustomEmbeddingTransformer("/Users/henrikjohn/Desktop/poincare_model_dim_3.pt")
  
  results <- PatientLevelPrediction::runPlp(
    plpData = plpData,
    outcomeId = 3,
    modelSettings = settings,
    analysisId = "Analysis_Poincare",
    analysisName = "Testing Deep Learning",
    populationSettings = populationSet,
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(),
    sampleSettings = PatientLevelPrediction::createSampleSettings(),
    featureEngineeringSettings = PatientLevelPrediction::createFeatureEngineeringSettings(),
    preprocessSettings = PatientLevelPrediction::createPreprocessSettings(),
    executeSettings = PatientLevelPrediction::createExecuteSettings(
      runSplitData = TRUE,
      runSampleData = FALSE,
      runfeatureEngineering = FALSE,
      runPreprocessData = FALSE,
      runModelDevelopment = TRUE,
      runCovariateSummary = FALSE
    ),
    saveDirectory = file.path(testLoc, "Poincare")
  )
  
  
  params <- defaultTransformer$param[[1]]

  expect_equal(params$numBlocks, 3)
  expect_equal(params$dimToken, 192)
  expect_equal(params$numHeads, 8)
  expect_equal(params$resDropout, 0.0)
  expect_equal(params$attDropout, 0.2)

  settings <- attr(defaultTransformer, "settings")

  expect_equal(settings$name, "defaultTransformer")
})