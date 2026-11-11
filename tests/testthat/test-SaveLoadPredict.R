# DPLP's side of the contract with PatientLevelPrediction: fitPlp() calls
# DPLP's fit function, savePlpModel() and loadPlpModel() round-trip its model
# directory, and predictPlp() calls its prediction function, which must
# reproduce the predictions made when the model was fitted. Splitting,
# preprocessing and evaluation are PLP's and are not tested here. Uses the
# simulated fixtures, so it needs Python but no database.

quickMlp <- function() {
  setMultiLayerPerceptron(
    numLayers = 1, sizeHidden = 8, dropout = 0, sizeEmbedding = 8,
    estimatorSettings = quickEstimator(), randomSample = 1
  )
}

# Fits with fitPlp() on the training data, saves and loads the model, and
# predicts the whole population with the loaded model.
roundTrip <- function(modelSettings, analysisId, trainData = simTrainData$Train,
                      plpData = simPlpData, population = simPopulation) {
  directory <- tempfile(analysisId)
  dir.create(directory)
  fitted <- quietly(PatientLevelPrediction::fitPlp(
    trainData = trainData,
    modelSettings = modelSettings,
    analysisId = analysisId,
    analysisPath = directory
  ))
  modelPath <- file.path(directory, "savedModel")
  PatientLevelPrediction::savePlpModel(fitted, modelPath)
  loaded <- PatientLevelPrediction::loadPlpModel(modelPath)
  list(
    modelPath = modelPath,
    fitted = fitted,
    predicted = quietly(PatientLevelPrediction::predictPlp(loaded, plpData, population))
  )
}

expectSamePredictions <- function(roundTrip) {
  expect_s3_class(roundTrip$fitted, "plpModel")
  train <- roundTrip$fitted$prediction[roundTrip$fitted$prediction$evaluationType == "Train", ]
  both <- merge(
    train[, c("rowId", "value")],
    roundTrip$predicted[, c("rowId", "value")],
    by = "rowId", suffixes = c("Fitted", "Loaded")
  )
  expect_equal(nrow(both), nrow(train))
  expect_equal(both$valueLoaded, both$valueFitted, tolerance = 1e-6)
}

realMlpSettings <- function() {
  settings <- setRealMLP(numLayers = 1, sizeHidden = 8, dropout = 0, sizeEmbedding = 8)
  settings$estimatorSettings$epochs <- 1L
  settings$estimatorSettings$batchSize <- 64L
  settings
}

# fitted once per run, shared by the tests below
delayedAssign("roundTrips", local({
  models <- list(
    ResNet = setResNet(
      numLayers = 1, sizeHidden = 8, hiddenFactor = 1, residualDropout = 0,
      hiddenDropout = 0, sizeEmbedding = 8, estimatorSettings = quickEstimator(),
      randomSample = 1
    ),
    MultiLayerPerceptron = quickMlp(),
    Transformer = setTransformer(
      numBlocks = 1, dimToken = 8, dimOut = 1, numHeads = 2, attDropout = 0,
      ffnDropout = 0, dimHidden = 16, estimatorSettings = quickEstimator()
    ),
    RealMLP = realMlpSettings()
  )
  mapply(roundTrip, models, names(models), SIMPLIFY = FALSE)
}))

test_that("each model predicts the same after saving and loading", {
  skip_if_no_python()
  for (name in names(roundTrips)) {
    expectSamePredictions(roundTrips[[name]])
  }
})

test_that("custom embedding models predict the same after saving and loading", {
  skip_if_no_python()
  # condition concepts get a vector from the embeddings file, which must still
  # exist when the loaded model predicts
  conceptIds <- simPlpData$covariateData$covariateRef %>%
    dplyr::filter(.data$analysisId == 102) %>%
    dplyr::pull("conceptId") %>%
    unique()
  embeddingFile <- withr::local_tempfile(fileext = ".pt")
  torch$manual_seed(42L)
  torch$save(list(
    concept_ids = torch$tensor(as.array(as.integer(conceptIds)), dtype = torch$long),
    embeddings = torch$randn(length(conceptIds), 4L)
  ), embeddingFile)
  customEmbedding <- setCustomEmbeddingModel(
    embeddingFilePath = embeddingFile,
    modelSettings = quickMlp()
  )
  fitted <- roundTrip(customEmbedding, "CustomEmbedding")
  expectSamePredictions(fitted)

  # fine-tuning builds the network with its custom embeddings again (#198)
  fineTuned <- setFinetuner(modelPath = fitted$modelPath, estimatorSettings = quickEstimator())
  expectSamePredictions(roundTrip(fineTuned, "FinetunedCustomEmbedding"))
})

test_that("fine-tuned models predict the same after saving and loading", {
  skip_if_no_python()
  # RealMLP had no reset_head() for the new output layer (#198)
  for (name in c("MultiLayerPerceptron", "RealMLP")) {
    fineTuned <- setFinetuner(
      modelPath = roundTrips[[name]]$modelPath,
      estimatorSettings = quickEstimator()
    )
    expectSamePredictions(roundTrip(fineTuned, paste0("Finetuned", name)))
  }
})

# simulated data turned into temporal sequences: every covariate gets one of
# 30 days before the index date
delayedAssign("simTemporal", local({
  plpData <- simulatedPlpData(n = 400)
  plpData$covariateData$covariates <- plpData$covariateData$covariates %>%
    dplyr::mutate(timeId = (.data$rowId + .data$covariateId) %% 30 + 1)
  plpData$covariateData$timeRef <- data.frame(
    timePart = "day", timeInterval = 1, sequenceStartDay = -30, sequenceEndDay = -1
  )
  population <- PatientLevelPrediction::createStudyPopulation(
    plpData, outcomeId = 3, populationSettings = simulatedPopulationSettings()
  )
  list(
    plpData = plpData,
    population = population,
    trainData = PatientLevelPrediction::splitData(
      plpData, population,
      PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42, nfold = 2)
    )
  )
}))

test_that("temporal models predict the same after saving and loading", {
  skip_if_no_python()
  temporalTransformer <- setTransformer(
    numBlocks = 1, dimToken = 8, dimOut = 1, numHeads = 2, attDropout = 0,
    ffnDropout = 0, dimHidden = 16, temporal = TRUE,
    temporalSettings = list(maxSequenceLength = 16, truncation = "tail", timeTokens = TRUE),
    estimatorSettings = quickEstimator()
  )
  expectSamePredictions(roundTrip(
    temporalTransformer, "TemporalTransformer",
    trainData = simTemporal$trainData$Train,
    plpData = simTemporal$plpData,
    population = simTemporal$population
  ))
})
