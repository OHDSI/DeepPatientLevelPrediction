# testing code (requires sequential branch of FeatureExtraction):
# rm(list = ls())
library(PatientLevelPrediction)
library(DeepPatientLevelPrediction)

# data(plpDataSimulationProfile)
# sampleSize <- 1e3
# plpData <- simulatePlpData(

#   plpDataSimulationProfile,
#   n = sampleSize
# )
#

plpData <- loadPlpData('~/cohorts/dementia/')

fix64bit <- function(plpData) {
  plpData$covariateData$covariateRef <- plpData$covariateData$covariateRef %>% 
    dplyr::mutate(covariateId=bit64::as.integer64(covariateId))
  plpData$covariateData$covariates <- plpData$covariateData$covariates %>%
    dplyr::mutate(rowId = as.integer(rowId),
                  covariateId = bit64::as.integer64(covariateId))
  plpData$cohorts <- plpData$cohorts %>% dplyr::mutate(rowId=as.integer(rowId))
  plpData$outcomes <- plpData$outcomes %>% dplyr::mutate(rowId = as.integer(rowId))
  
  return(plpData)
}

#downsample for speed
# plpData$cohorts <- plpData$cohorts[sample.int(nrow(plpData$cohorts), 1e5),]

populationSet <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = F, 
  riskWindowStart = 1, 
  riskWindowEnd = 365*5)


modelSettings <- setResNet(numLayers = 4L, 
                           sizeHidden = 256L, 
                           hiddenFactor = 2L,
                           residualDropout = 0.2, 
                           hiddenDropout = 0.2, 
                           sizeEmbedding = 128L, 
                           estimatorSettings = setEstimator(learningRate= "auto",
                                                            weightDecay = 1e-06,
                                                            device='cuda:0',
                                                            batchSize=128L,
                                                            epochs=3L,
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
  outcomeId = 2,
  modelSettings = modelSettings,
  analysisId = 'Test',
  analysisName = 'Testing DeepPlp',
  populationSettings = populationSet,
  splitSettings = createDefaultSplitSetting(),
  sampleSettings = createSampleSettings("underSample"),  # none
  featureEngineeringSettings = createFeatureEngineeringSettings(), # none
  preprocessSettings = createPreprocessSettings(),
  logSettings = createLogSettings(verbosity='TRACE'),
  executeSettings = createExecuteSettings(
    runSplitData = T,
    runSampleData = T,
    runfeatureEngineering = F,
    runPreprocessData = T,
    runModelDevelopment = T,
    runCovariateSummary = F
  ),
  saveDirectory = '~/test/new_plp/'
)


