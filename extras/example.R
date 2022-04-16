# testing code (requires sequential branch of FeatureExtraction):
# rm(list = ls())
library(FeatureExtraction)
library(PatientLevelPredictionArrow)
library(DeepPatientLevelPrediction)

arrow <- T
data(plpDataSimulationProfile)
sampleSize <- 1e4
plpData <- simulatePlpData(
  plpDataSimulationProfile,
  n = sampleSize
)


populationSet <- PatientLevelPredictionArrow::createStudyPopulationSettings(
  requireTimeAtRisk = F, 
  riskWindowStart = 1, 
  riskWindowEnd = 365)


# modelSettings <- PatientLevelPrediction::setGradientBoostingMachine(ntrees = 100, nthread = 16, 
#                                             earlyStopRound = 25, maxDepth = 6,
#                                             minChildWeight = 1, learnRate = 0.3,
#                                             seed = 42)

# modelSettings <- PatientLevelPredictionArrow::setLassoLogisticRegression()

modelSettings <- DeepPatientLevelPrediction::setTabNetTorch(device='cuda:0', randomSamples = 1,
                                                            batchSize = 32)

if (arrow) {
  res2 <- runPlp(
  plpData = plpData,
  outcomeId = 3,
  modelSettings = modelSettings,
  analysisId = 'Test',
  analysisName = 'Testing ARrow',
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
  saveDirectory = '~/test/arrow_new_plp/'
)
} else {
  library(PatientLevelPrediction)
  res2 <- PatientLevelPrediction::runPlp(
  plpData = plpData,
  outcomeId = 3,
  modelSettings = modelSettings,
  analysisId = 'Test',
  analysisName = 'Testing Original',
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
}

