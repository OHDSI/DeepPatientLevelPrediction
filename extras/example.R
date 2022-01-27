# testing code (requires sequential branch of FeatureExtraction):
rm(list = ls())
library(FeatureExtraction)
library(PatientLevelPrediction)
library(DeepPatientLevelPrediction)

temp <- F

data(plpDataSimulationProfile)
sampleSize <- 2000
plpData <- simulatePlpData(
  plpDataSimulationProfile,
  n = sampleSize
)
population <- createStudyPopulation(
  plpData,
  outcomeId = 2,
  binary = TRUE,
  firstExposureOnly = FALSE,
  washoutPeriod = 0,  
  removeSubjectsWithPriorOutcome = FALSE,
  priorOutcomeLookback = 99999,
  requireTimeAtRisk = FALSE,
  minTimeAtRisk = 0,
  riskWindowStart = 0,
  riskWindowEnd = 365,
  verbosity = "INFO"
)


resSet <- setResNet(numLayers=5, sizeHidden=256, hiddenFactor=2,
                    residualDropout=c(0.1), 
                    hiddenDropout=c(0.1),
                    normalization='BatchNorm', activation= 'RelU',
                    sizeEmbedding=64, weightDecay=c(1e-6),
                    learningRate=c(3e-4), seed=42, hyperParamSearch='random',
                    randomSample=1, 
                    device='cuda:0', 
                    batchSize=512, 
                    epochs=1)

debug(trainResNet)
res2 <- runPlp(population = population, 
              plpData = plpData, 
              nfold = 3,
              modelSettings = resSet,
              savePlpData = F, 
              savePlpResult = F, 
              savePlpPlots = F, 
              saveEvaluation = F)