# Every model must predict the same after a full PatientLevelPrediction round
# trip: runPlp(), savePlpModel(), loadPlpModel() and predictPlp(). Simulated
# data keeps this offline; it needs Python but no database.

quickEstimator <- function() {
  setEstimator(learningRate = 1e-3, batchSize = 64, epochs = 1, seed = 42, device = "cpu")
}

# Fits with runPlp(), saves and loads the model, and returns the predictions
# for the whole population from the fitted and the loaded model, plus the
# test predictions that runPlp() stored.
roundTrip <- function(plpData, modelSettings, directory, analysisId) {
  result <- suppressMessages(PatientLevelPrediction::runPlp(
    plpData = plpData,
    outcomeId = 3,
    analysisId = analysisId,
    populationSettings = simulatedPopulationSettings(),
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(
      splitSeed = 42, nfold = 2
    ),
    modelSettings = modelSettings,
    executeSettings = PatientLevelPrediction::createExecuteSettings(
      runSplitData = TRUE, runModelDevelopment = TRUE
    ),
    saveDirectory = directory
  ))
  modelPath <- file.path(directory, analysisId, "savedModel")
  PatientLevelPrediction::savePlpModel(result$model, modelPath)
  loaded <- PatientLevelPrediction::loadPlpModel(modelPath)
  population <- PatientLevelPrediction::createStudyPopulation(
    plpData, outcomeId = 3, populationSettings = simulatedPopulationSettings()
  )
  list(
    modelPath = modelPath,
    stored = result$prediction[result$prediction$evaluationType == "Test", ],
    fitted = suppressMessages(PatientLevelPrediction::predictPlp(result$model, plpData, population)),
    loaded = suppressMessages(PatientLevelPrediction::predictPlp(loaded, plpData, population))
  )
}

expectSamePredictions <- function(predictions) {
  expect_equal(predictions$loaded$value, predictions$fitted$value)
  # the loaded model reproduces the test predictions made during training
  test <- merge(
    predictions$stored[, c("rowId", "value")],
    predictions$loaded[, c("rowId", "value")],
    by = "rowId", suffixes = c("Stored", "Loaded")
  )
  expect_equal(nrow(test), nrow(predictions$stored))
  expect_equal(test$valueLoaded, test$valueStored, tolerance = 1e-6)
}

test_that("each model predicts the same after saving and loading", {
  skip_if_no_python()
  plpData <- simulatedPlpData(n = 400)
  directory <- withr::local_tempdir()
  realMlp <- setRealMLP(numLayers = 1, sizeHidden = 8, dropout = 0, sizeEmbedding = 8)
  realMlp$estimatorSettings$epochs <- 1L
  realMlp$estimatorSettings$batchSize <- 64L
  models <- list(
    ResNet = setResNet(
      numLayers = 1, sizeHidden = 8, hiddenFactor = 1, residualDropout = 0,
      hiddenDropout = 0, sizeEmbedding = 8, estimatorSettings = quickEstimator(),
      randomSample = 1
    ),
    MultiLayerPerceptron = setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 8, dropout = 0, sizeEmbedding = 8,
      estimatorSettings = quickEstimator(), randomSample = 1
    ),
    Transformer = setTransformer(
      numBlocks = 1, dimToken = 8, dimOut = 1, numHeads = 2, attDropout = 0,
      ffnDropout = 0, dimHidden = 16, estimatorSettings = quickEstimator()
    ),
    RealMLP = realMlp
  )
  for (name in names(models)) {
    predictions <- roundTrip(plpData, models[[name]], directory, name)
    expectSamePredictions(predictions)
  }
})

test_that("custom embedding models predict the same after saving and loading", {
  skip_if_no_python()
  plpData <- simulatedPlpData(n = 400)
  directory <- withr::local_tempdir()

  # condition concepts get a vector from the embeddings file, which must still
  # exist when the loaded model predicts
  conceptIds <- plpData$covariateData$covariateRef %>%
    dplyr::filter(.data$analysisId == 102) %>%
    dplyr::pull("conceptId") %>%
    unique()
  embeddingFile <- file.path(directory, "embeddings.pt")
  torch$save(list(
    concept_ids = torch$tensor(as.array(as.integer(conceptIds)), dtype = torch$long),
    embeddings = torch$randn(length(conceptIds), 4L)
  ), embeddingFile)
  customEmbedding <- setCustomEmbeddingModel(
    embeddingFilePath = embeddingFile,
    modelSettings = setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 8, dropout = 0, sizeEmbedding = 8,
      estimatorSettings = quickEstimator(), randomSample = 1
    )
  )
  expectSamePredictions(roundTrip(plpData, customEmbedding, directory, "CustomEmbedding"))
})

test_that("fine-tuned models predict the same after saving and loading", {
  skip_if_no_python()
  plpData <- simulatedPlpData(n = 400)
  directory <- withr::local_tempdir()
  base <- roundTrip(
    plpData,
    setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 8, dropout = 0, sizeEmbedding = 8,
      estimatorSettings = quickEstimator(), randomSample = 1
    ),
    directory,
    "Base"
  )
  fineTuned <- setFinetuner(modelPath = base$modelPath, estimatorSettings = quickEstimator())
  expectSamePredictions(roundTrip(plpData, fineTuned, directory, "Finetuner"))
})
