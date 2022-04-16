# testing code (requires sequential branch of FeatureExtraction):
# rm(list = ls())
library(FeatureExtraction)
library(PatientLevelPrediction)
library(DeepPatientLevelPrediction)

data(plpDataSimulationProfile)
sampleSize <- 1e4
plpData <- simulatePlpData(
  plpDataSimulationProfile,
  n = sampleSize
)


populationSet <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = F, 
  riskWindowStart = 1, 
  riskWindowEnd = 365)


modelSettings <- setResNet(numLayers = 2, sizeHidden = 64, hiddenFactor = 1,
                          residualDropout = 0, hiddenDropout = 0.2, normalization = 'BatchNorm',
                          activation = 'RelU', sizeEmbedding = 64, weightDecay = 1e-6,
                          learningRate = 3e-4, seed = 42, hyperParamSearch = 'random',
                          randomSample = 1, device = 'cuda:0',batchSize = 32,epochs = 1)

res2 <- PatientLevelPrediction::runPlp(
plpData = plpData,
outcomeId = 3,
modelSettings = modelSettings,
analysisId = 'Test',
analysisName = 'Testing DeepPlp',
populationSettings = populationSet,
splitSettings = createDefaultSplitSetting(),
sampleSettings = createSampleSettings(),  # none
featureEngineeringSettings = createFeatureEngineeringSettings(), # none
preprocessSettings = createPreprocessSettings(),
logSettings = createLogSettings(verbosity='TRACE'),
executeSettings = createExecuteSettings(
  runSplitData = T,
  runSampleData = F,
  runfeatureEngineering = F,
  runPreprocessData = T,
  runModelDevelopment = T,
  runCovariateSummary = T
),
saveDirectory = '~/test/new_plp/'
)


