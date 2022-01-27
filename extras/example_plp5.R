# testing code (requires sequential branch of FeatureExtraction):
rm(list = ls())
library(FeatureExtraction)
library(PatientLevelPrediction)
library(DeepPatientLevelPrediction)
connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

temp <- F

covSet <- createCovariateSettings(useDemographicsGender = T, 
                                  useDemographicsAge = T, 
                                  useDemographicsRace = T,
                                  useDemographicsEthnicity = T, 
                                  useDemographicsAgeGroup = T,
                                  useConditionGroupEraLongTerm = T, 
                                  useDrugEraStartLongTerm  = T, 
                                  endDays = -1
)

if(temp){
covSetT <- createTemporalSequenceCovariateSettings(useDemographicsGender = T, 
                                                  useDemographicsAge = T, 
                                                  useDemographicsRace = T,
                                                  useDemographicsEthnicity = T, 
                                                  useDemographicsAgeGroup = T,
                                                  useConditionEraGroupStart = T, 
                                                  useDrugEraStart = T, 
                                                  timePart = 'month', 
                                                  timeInterval = 1, 
                                                  sequenceEndDay = -1, 
                                                  sequenceStartDay = -365*5)
}


databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
  connectionDetails = connectionDetails, 
  cdmDatabaseSchema = "main",
  cohortDatabaseSchema = "main", 
  cohortTable = "cohort", 
  cohortId = 4, 
  outcomeIds = 3,
  outcomeDatabaseSchema = "main", 
  outcomeTable =  "cohort", 
  cdmDatabaseName = 'eunomia'
)

restrictPlpDataSettings <- PatientLevelPrediction::createRestrictPlpDataSettings(
  firstExposureOnly = T, 
  washoutPeriod = 365
)
  
plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails, 
  restrictPlpDataSettings = restrictPlpDataSettings,
  covariateSettings = covSet
)

if(temp){
  plpDataT <- PatientLevelPrediction::getPlpData(  
    databaseDetails = databaseDetails, 
    restrictPlpDataSettings = restrictPlpDataSettings,
    covariateSettings = covSetT
  )
}


populationSet <- PatientLevelPrediction::createStudyPopulationSettings(
                                                            requireTimeAtRisk = F, 
                                                            riskWindowStart = 1, 
                                                            riskWindowEnd = 365
  )

# code to train models
deepset <- setDeepNNTorch(units=list(c(128, 64), 128), layer_dropout=c(0.2),
                          lr =c(1e-4), decay=c(1e-5), outcome_weight = c(1.0), batch_size = c(100), 
                          epochs= c(5),  seed=NULL  )


#debug(fitDeepNNTorch)
# res <- runPlp(population = population, 
#               plpData = plpData, 
#               nfold = 3,
#               modelSettings = deepset, 
#               savePlpData = F, 
#               savePlpResult = F, 
#               savePlpPlots = F, 
#               saveEvaluation = F)
# 


resSet <- setResNet_plp5(
  numLayers = list(5), 
  sizeHidden = list(256),
  hiddenFactor = list(2),
    residualDropout = list(0.1), 
    hiddenDropout = list(0.1),
    normalization = list('BatchNorm'), 
    activation = list('RelU'),
    sizeEmbedding = list(64), 
    weightDecay = list(1e-6),
    learningRate = list(3e-4), 
    seed = 42, 
    hyperParamSearch = 'random',
    randomSample = 1, 
    #device='cuda:0', 
    batchSize = 128, 
    epochs = 10
  )


res2 <- runPlp(
  plpData = plpData, 
  outcomeId = 3, 
  modelSettings = resSet,
  analysisId = 'ResNet', 
  analysisName = 'Testing Deep Learning', 
  populationSettings = populationSet, 
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting(), 
  sampleSettings = PatientLevelPrediction::createSampleSettings(),  # none 
  featureEngineeringSettings = PatientLevelPrediction::createFeatureEngineeringSettings(), # none 
  preprocessSettings = PatientLevelPrediction::createPreprocessSettings(), 
  executeSettings = PatientLevelPrediction::createExecuteSettings(
    runSplitData = T, 
    runSampleData = F, 
    runfeatureEngineering = F, 
    runPreprocessData = T, 
    runModelDevelopment = T, 
    runCovariateSummary = F
    ), 
  saveDirectory = 'D:/testing/Deep'
  )


##predict.customLibrary(libraryName, predictionFunction, inputList){
##  libraryName <- 'PatientLevelPrediction'
##  predictionFunction <- "createStudyPopulation"
##  predictFun <- get(predictionFunction, envir = rlang::search_envs()[grep(paste0('package:', libraryName), search())][[1]])
##  
##  prediction <- do.call(predictFun, inputList)
##  return(prediction)
##}
