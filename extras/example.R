# testing code (requires sequential branch of FeatureExtraction):
# rm(list = ls())
library(PatientLevelPrediction)
library(DeepPatientLevelPrediction)

data(plpDataSimulationProfile)
sampleSize <- 1e3
plpData <- simulatePlpData(
   plpDataSimulationProfile,
   n = sampleSize
 )
#
# plpData <- loadPlpData('~/cohorts/ComparisonStudy/DementiaResults/Dementia/')
# 
# fix64bit <- function(plpData) {
#   plpData$covariateData$covariateRef <- plpData$covariateData$covariateRef |>
#     dplyr::mutate(covariateId=bit64::as.integer64(covariateId))
#   plpData$covariateData$covariates <- plpData$covariateData$covariates |>
#     dplyr::mutate(rowId = as.integer(rowId),
#                   covariateId = bit64::as.integer64(covariateId))
#   plpData$cohorts <- plpData$cohorts |> dplyr::mutate(rowId=as.integer(rowId))
#   plpData$outcomes <- plpData$outcomes |> dplyr::mutate(rowId = as.integer(rowId))
#   
#   return(plpData)
# }

# plpData <- fix64bit(plpData)
#downsample for speed
# plpData$cohorts <- plpData$cohorts[sample.int(nrow(plpData$cohorts), 1e5),]


populationSet <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = F, 
  riskWindowStart = 1, 
  riskWindowEnd = 365*5)

# 
# modelSettings <- setDefaultTransformer(estimatorSettings = setEstimator(
#   learningRate = "auto",
#   batchSize=64L,
#   device="cuda:0",
#   epochs = 10L
# ))

modelSettings <- setDefaultResNet(estimatorSettings = setEstimator(
  learningRate = "auto",
  weightDecay = 1e-06,
  device="cuda:0",
  batchSize=128L,
  epochs=50L,
  seed=42
))

modelSettings <- setResNet(numLayers = c(1L, 2L),
                           sizeHidden = 72L,
                           hiddenFactor = 1L,
                           residualDropout = 0.0,
                           hiddenDropout = 0.0,
                           sizeEmbedding = 64L,
                           estimatorSettings = setEstimator(
                             learningRate = 3e-4,
                             batchSize = 128L,
                             epochs = 10L,
                             device = "cpu",
                             seed = 42
                           ),
                           randomSample = 2)

res2 <- PatientLevelPrediction::runPlp(
  plpData = plpData,
  outcomeId = unique(plpData$outcomes$outcomeId)[[1]],
  modelSettings = modelSettings,
  analysisId = 'Test',
  analysisName = 'Testing DeepPlp',
  populationSettings = populationSet,
  splitSettings = createDefaultSplitSetting(splitSeed = 123),
  sampleSettings = createSampleSettings("underSample"),  # none
  featureEngineeringSettings = createFeatureEngineeringSettings(), # none
  preprocessSettings = createPreprocessSettings(normalize = F),
  logSettings = createLogSettings(verbosity='TRACE'),
  executeSettings = createExecuteSettings(
    runSplitData = T,
    runSampleData = T,
    runfeatureEngineering = F,
    runPreprocessData = T,
    runModelDevelopment = T,
    runCovariateSummary = F
  ),
  saveDirectory = '~/test/resnet/'
)


