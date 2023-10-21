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


populationSet <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = F, 
  riskWindowStart = 1, 
  riskWindowEnd = 365*5)

# 
# modelSettings <- setDefaultTransformer(estimatorSettings = setEstimator(
#   learningRate = "auto",
#   batchSize=64L,
#   epochs = 10L
# ))

# modelSettings <- setDefaultResNet(estimatorSettings = setEstimator(
#   learningRate = "auto",
#   weightDecay = 1e-06,
#   device="cuda:0",
#   batchSize=128L,
#   epochs=50L,
#   seed=42
# ))

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
                           randomSample = 2,
                           randomSampleSeed = 1)

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
  saveDirectory = '~/deep_plp_test/resnet/'
)


