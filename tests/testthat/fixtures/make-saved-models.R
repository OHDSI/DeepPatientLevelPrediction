# Writes the saved-model fixtures that test-SavedModels.R loads: small models
# saved by this version of the package, the simulated data they were fitted
# on, and the predictions they made. Install the package first, then run from
# the package root, with the Python environment set up:
#
#   R CMD INSTALL .
#   Rscript tests/testthat/fixtures/make-saved-models.R
#
# The installed package is used rather than pkgload::load_all(), because
# load_all() keeps source references in functions, which makes the saved
# settings about a hundred times larger than users' saved models.
#
# The fixtures must not be regenerated just to make that test pass: models
# that users saved with this version have to keep predicting the same.
library(DeepPatientLevelPrediction)
sys.source("tests/testthat/helper-mock.R", envir = environment())
sys.source("tests/testthat/helper-fixtures.R", envir = environment())

version <- as.character(utils::packageVersion("DeepPatientLevelPrediction"))
root <- file.path("tests/testthat/fixtures", paste0("saved-models-", version))
unlink(root, recursive = TRUE)
dir.create(root, recursive = TRUE)

plpData <- simulatedPlpData(n = 200)
PatientLevelPrediction::savePlpData(plpData, file.path(root, "plpData"))
population <- PatientLevelPrediction::createStudyPopulation(
  plpData, outcomeId = 3, populationSettings = simulatedPopulationSettings()
)
trainData <- PatientLevelPrediction::splitData(
  plpData, population,
  PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42, nfold = 2)
)
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
  analysisPath <- tempfile(name)
  dir.create(analysisPath)
  fitted <- PatientLevelPrediction::fitPlp(
    trainData$Train, models[[name]], analysisId = name, analysisPath = analysisPath
  )
  modelPath <- file.path(root, name)
  PatientLevelPrediction::savePlpModel(fitted, modelPath)
  predictions <- PatientLevelPrediction::predictPlp(fitted, plpData, population)
  utils::write.csv(predictions[, c("rowId", "value")], file.path(modelPath, "predictions.csv"), row.names = FALSE)
  # .Rbuildignore leaves out *.pt files, so store the weights under another name
  file.rename(
    file.path(modelPath, "model", "DeepEstimatorModel.pt"),
    file.path(modelPath, "model", "DeepEstimatorModel.pt.bin")
  )
}
