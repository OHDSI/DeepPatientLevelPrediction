# testing code (requires sequential branch of FeatureExtraction):
# rm(list = ls())
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

 
modelSettings <- setResNet(numLayers = 2, 
                           sizeHidden = 64, 
                           hiddenFactor = 1,
                          residualDropout = 0, 
                          hiddenDropout = 0.2 , 
                          sizeEmbedding = 512, 
                          estimatorSettings = setEstimator(learningRate= 3e-4,
                                                           weightDecay = 1e-06,
                                                           device='cuda:0',
                                                           batchSize=128,
                                                           epochs=1,
                                                           seed=42),
                          hyperParamSearch = 'random',
                          randomSample = 1)

# modelSettings <- setTransformer(numBlocks=1, dimToken = 33, dimOut = 1, numHeads = 3,
#                                 attDropout = 0.2, ffnDropout = 0.2, resDropout = 0,
#                                 dimHidden = 8, batchSize = 32, hyperParamSearch = 'random',
#                                 weightDecay = 1e-6, learningRate = 3e-4, epochs = 10,
#                                 device = 'cuda:0', randomSamples = 1, seed = 42)

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


