
test_that("Poincare Transformer works", {
  concept_ids <- torch$Tensor(plpData$covariateData$covariateRef %>% dplyr::filter(
    analysisId == 210) %>% dplyr::pull("conceptId"))$long()
  
  customEmbeddings <- list("concept_ids" = concept_ids, "embeddings" = torch$randn(3L, concept_ids$shape[0]))
  torch$save(customEmbeddings, file.path(testLoc, "custom_embeddings.pt"))
  settings <- setCustomEmbeddingModel(embeddingFilePath = file.path(testLoc, "custom_embeddings.pt"),
                                      modelSettings = setTransformer(
                                        numBlocks = 1,
                                        dimToken = 6,
                                        numHeads = 1,
                                        dimHidden = 12,
                                        estimatorSettings = setEstimator(learningRate = 1e-3,
                                                                         epochs = 1,
                                                                         device = "cpu")
                                      ))
  
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
  
  
})