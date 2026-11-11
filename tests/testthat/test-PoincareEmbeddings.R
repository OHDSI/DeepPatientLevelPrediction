skip_if_no_python()

# condition concepts of the simulated data
conceptIds <- torch$tensor(simPlpData$covariateData$covariateRef %>%
  dplyr::filter(analysisId == 102) %>%
  dplyr::pull("conceptId") %>%
  unique(), dtype = torch$long)

# points of the Poincare ball: each coordinate is below 0.5 in absolute value,
# so every norm is below sqrt(3) / 2
customEmbeddings <- list(
  "concept_ids" = conceptIds,
  "embeddings" = torch$rand(conceptIds$shape[0], 3L) - 0.5
)
torch$save(customEmbeddings, file.path(tempdir(), "custom_embeddings.pt"))

test_that("Poincare Transformer works", {
  settings <- setCustomEmbeddingModel(
    embeddingFilePath = file.path(tempdir(), "custom_embeddings.pt"),
    modelSettings = setTransformer(
      numBlocks = 1,
      dimToken = 6,
      numHeads = 1,
      dimHidden = 12,
      estimatorSettings = setEstimator(
        learningRate = 1e-3,
        epochs = 1,
        device = "cpu"
      )
    ),
    embeddingsClass = "PoincareEmbeddings"
  )

  results <- PatientLevelPrediction::runPlp(
    plpData = simPlpData,
    outcomeId = 3,
    modelSettings = settings,
    analysisId = "Analysis_Poincare",
    analysisName = "Testing Deep Learning",
    populationSettings = simulatedPopulationSettings(),
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42, nfold = 2),
    sampleSettings = PatientLevelPrediction::createSampleSettings(),
    featureEngineeringSettings = PatientLevelPrediction::createFeatureEngineeringSettings(),
    preprocessSettings = PatientLevelPrediction::createPreprocessSettings(),
    executeSettings = PatientLevelPrediction::createExecuteSettings(
      runSplitData = TRUE,
      runSampleData = FALSE,
      runFeatureEngineering = FALSE,
      runPreprocessData = FALSE,
      runModelDevelopment = TRUE,
      runCovariateSummary = FALSE
    ),
    saveDirectory = file.path(tempdir(), "Poincare")
  )

  model <- torch$load(file.path(results$model$model, "DeepEstimatorModel.pt"),
    weights_only = FALSE
  )

  # get custom embeddings and verify they are same as the ones created above
  customModelEmbeddings <- model$model_state_dict$embedding.custom_embeddings.weight
  conceptsInData <- results$model$covariateImportance %>%
    dplyr::filter(analysisId == 102) %>%
    dplyr::pull(.data$conceptId)
  embeddingsMask <- torch$isin(
    customEmbeddings[["concept_ids"]],
    torch$tensor(conceptsInData, dtype = torch$long)
  )
  embeddingsInData <- customEmbeddings[["embeddings"]][embeddingsMask]

  expect_equal(customModelEmbeddings$shape[0] - 1, length(conceptsInData)) # -1 for padding embedding
  expect_equal(
    embeddingsInData$mean()$item(),
    customModelEmbeddings[1:customModelEmbeddings$shape[0], ]$mean()$item()
  )
})

test_that("ResNet with poincare works", {
  settings <- setCustomEmbeddingModel(
    embeddingFilePath = file.path(tempdir(), "custom_embeddings.pt"),
    modelSettings = setResNet(
      numLayers = 1,
      sizeEmbedding = 6,
      sizeHidden = 12,
      hiddenFactor = 1,
      residualDropout = 0.0,
      hiddenDropout = 0.0,
      estimatorSettings = setEstimator(
        learningRate = 1e-3,
        epochs = 1,
        device = "cpu"
      ),
      randomSample = 1
    ),
    embeddingsClass = "PoincareEmbeddings"
  )

  results <- PatientLevelPrediction::runPlp(
    plpData = simPlpData,
    outcomeId = 3,
    modelSettings = settings,
    analysisId = "Analysis_Poincare",
    analysisName = "Testing Deep Learning",
    populationSettings = simulatedPopulationSettings(),
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42, nfold = 2),
    sampleSettings = PatientLevelPrediction::createSampleSettings(),
    featureEngineeringSettings = PatientLevelPrediction::createFeatureEngineeringSettings(),
    preprocessSettings = PatientLevelPrediction::createPreprocessSettings(),
    executeSettings = PatientLevelPrediction::createExecuteSettings(
      runSplitData = TRUE,
      runSampleData = FALSE,
      runFeatureEngineering = FALSE,
      runPreprocessData = FALSE,
      runModelDevelopment = TRUE,
      runCovariateSummary = FALSE
    ),
    saveDirectory = file.path(tempdir(), "PoincareResNet")
  )

  model <- torch$load(file.path(results$model$model, "DeepEstimatorModel.pt"),
    weights_only = FALSE
  )

  # get custom embeddings and verify they are same as the ones created above
  customModelEmbeddings <- model$model_state_dict$embedding.custom_embeddings.weight
  conceptsInData <- results$model$covariateImportance %>%
    dplyr::filter(
      analysisId == 102
    ) %>%
    dplyr::pull("conceptId")
  embeddingsMask <- torch$isin(customEmbeddings[["concept_ids"]], torch$tensor(conceptsInData, dtype = torch$long))
  embeddingsInData <- customEmbeddings[["embeddings"]][embeddingsMask]

  expect_equal(customModelEmbeddings$shape[0] - 1, length(conceptsInData)) # -1 for padding embedding
  expect_equal(embeddingsInData$mean()$item(), customModelEmbeddings[1:customModelEmbeddings$shape[0], ]$mean()$item())
})

test_that("MLP with poincare works", {
  settings <- setCustomEmbeddingModel(
    embeddingFilePath = file.path(tempdir(), "custom_embeddings.pt"),
    modelSettings = setMultiLayerPerceptron(
      numLayers = 1,
      sizeEmbedding = 6,
      sizeHidden = 12,
      dropout = 0.0,
      estimatorSettings = setEstimator(
        learningRate = 1e-3,
        epochs = 1,
        device = "cpu"
      ),
      randomSample = 1
    ),
    embeddingsClass = "PoincareEmbeddings"
  )

  results <- PatientLevelPrediction::runPlp(
    plpData = simPlpData,
    outcomeId = 3,
    modelSettings = settings,
    analysisId = "Analysis_Poincare",
    analysisName = "Testing Deep Learning",
    populationSettings = simulatedPopulationSettings(),
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42, nfold = 2),
    sampleSettings = PatientLevelPrediction::createSampleSettings(),
    featureEngineeringSettings = PatientLevelPrediction::createFeatureEngineeringSettings(),
    preprocessSettings = PatientLevelPrediction::createPreprocessSettings(),
    executeSettings = PatientLevelPrediction::createExecuteSettings(
      runSplitData = TRUE,
      runSampleData = FALSE,
      runFeatureEngineering = FALSE,
      runPreprocessData = FALSE,
      runModelDevelopment = TRUE,
      runCovariateSummary = FALSE
    ),
    saveDirectory = file.path(tempdir(), "PoincareMLP")
  )

  model <- torch$load(file.path(results$model$model, "DeepEstimatorModel.pt"),
    weights_only = FALSE
  )

  # get custom embeddings and verify they are same as the ones created above
  customModelEmbeddings <- model$model_state_dict$embedding.custom_embeddings.weight
  conceptsInData <- results$model$covariateImportance %>%
    dplyr::filter(analysisId == 102) %>%
    dplyr::pull("conceptId")
  embeddingsMask <- torch$isin(customEmbeddings[["concept_ids"]], torch$tensor(conceptsInData, dtype = torch$long))
  embeddingsInData <- customEmbeddings[["embeddings"]][embeddingsMask]

  expect_equal(customModelEmbeddings$shape[0] - 1, length(conceptsInData)) # -1 for padding embedding
  expect_equal(embeddingsInData$mean()$item(), customModelEmbeddings[1:customModelEmbeddings$shape[0], ]$mean()$item())
})
