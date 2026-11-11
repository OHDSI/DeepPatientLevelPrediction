# DPLP's side of the contract with PatientLevelPrediction: fitPlp() calls
# DPLP's fit function, savePlpModel() and loadPlpModel() round-trip its model
# directory, and predictPlp() calls its prediction function, which must
# reproduce the predictions made when the model was fitted. Splitting,
# preprocessing and evaluation are PLP's and are not tested here. Uses the
# simulated fixtures, so it needs Python but no database.

quickEstimator <- function() {
  setEstimator(learningRate = 1e-3, batchSize = 64, epochs = 1, seed = 42, device = "cpu")
}

quickMlp <- function() {
  setMultiLayerPerceptron(
    numLayers = 1, sizeHidden = 8, dropout = 0, sizeEmbedding = 8,
    estimatorSettings = quickEstimator(), randomSample = 1
  )
}

# Fits with fitPlp() on the simulated training data, saves and loads the
# model, and predicts the whole simulated population with the loaded model.
roundTrip <- function(modelSettings, analysisId) {
  directory <- tempfile(analysisId)
  dir.create(directory)
  fitted <- quietly(PatientLevelPrediction::fitPlp(
    trainData = simTrainData$Train,
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
    predicted = quietly(PatientLevelPrediction::predictPlp(loaded, simPlpData, simPopulation))
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
  expectSamePredictions(roundTrip(customEmbedding, "CustomEmbedding"))
})

test_that("fine-tuned models predict the same after saving and loading", {
  skip_if_no_python()
  fineTuned <- setFinetuner(
    modelPath = roundTrips$MultiLayerPerceptron$modelPath,
    estimatorSettings = quickEstimator()
  )
  expectSamePredictions(roundTrip(fineTuned, "Finetuner"))
})
