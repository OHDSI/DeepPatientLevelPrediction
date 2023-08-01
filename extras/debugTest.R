library(PatientLevelPrediction)
library(DeepPatientLevelPrediction)


data(plpDataSimulationProfile)
sampleSize <- 1e4
plpData <- simulatePlpData(
  plpDataSimulationProfile,
  n = sampleSize
)

populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = F,
  riskWindowStart = 1,
  riskWindowEnd = 365)

modelSettings <- setResNet(numLayers = 4, sizeHidden = 512, hiddenFactor = 2, residualDropout = 0.2,
                           hiddenDropout = 0.2, sizeEmbedding = 32, weightDecay = 0, learningRate = 3e-4,
                           randomSample = 1, batchSize = 1024, device = 'cuda', epochs=50)

# modelSettings <- setMultiLayerPerceptron(batchSize=2056, device='cuda',
#                                          epochs =50)

plpResults <-  PatientLevelPrediction::runPlp(
    plpData = plpData, 
    outcomeId = 2, 
    modelSettings = modelSettings,
    analysisId = 'MLP without posWeight', 
    analysisName = 'Testing Deep Learning', 
    populationSettings = populationSettings, 
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed=42), 
    sampleSettings = PatientLevelPrediction::createSampleSettings(),  # none 
    featureEngineeringSettings = PatientLevelPrediction::createFeatureEngineeringSettings(), # none 
    preprocessSettings = PatientLevelPrediction::createPreprocessSettings(), 
    logSettings = PatientLevelPrediction::createLogSettings(verbosity="TRACE"),
    executeSettings = PatientLevelPrediction::createExecuteSettings(
      runSplitData = T,
      runSampleData = T, 
      runfeatureEngineering = F, 
      runPreprocessData = T, 
      runModelDevelopment = T, 
      runCovariateSummary = F
    ), 
    saveDirectory = '~/projects/dementia/results/mlp_no_pos_weight')
