
test_that("Poincare Transformer works", {
  conceptIds <- torch$tensor(plpData$covariateData$covariateRef %>% dplyr::filter(
    analysisId == 210) %>% dplyr::pull("conceptId"), dtype = torch$long)
  
  customEmbeddings <- list("concept_ids" = conceptIds, "embeddings" = torch$randn(conceptIds$shape[0], 3L))
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
                                      ),
                                      embeddingsClass = "PoincareEmbeddings")
  
  results <- PatientLevelPrediction::runPlp(
    plpData = plpData,
    outcomeId = 3,
    modelSettings = settings,
    analysisId = "Analysis_Poincare",
    analysisName = "Testing Deep Learning",
    populationSettings = populationSet,
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42, nfold = 2),
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
  
  model <- torch$load(file.path(results$model$model, "DeepEstimatorModel.pt"),
                      weights_only = FALSE)
  
  # get custom embeddings and verify they are same as the ones created above
  custom_embeddings <- model$model_state_dict$embedding.custom_embeddings.weight
  conceptsInData <- results$model$covariateImportance %>% dplyr::filter(
    analysisId == 210) %>% dplyr::pull("conceptId")
  embeddingsMask <- torch$isin(customEmbeddings[["concept_ids"]], torch$tensor(conceptsInData, dtype = torch$long))
  embeddingsInData <- customEmbeddings[["embeddings"]][embeddingsMask]
    
  expect_equal(custom_embeddings$shape[0] - 1, length(conceptsInData)) # -1 for padding embedding
  expect_equal(embeddingsInData$mean()$item(),  custom_embeddings[1:custom_embeddings$shape[0],]$mean()$item())
})