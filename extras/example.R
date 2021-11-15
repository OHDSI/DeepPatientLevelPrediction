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

plpData <- PatientLevelPrediction::getPlpData(connectionDetails = connectionDetails, 
                                               cdmDatabaseSchema = "main",
                                               cohortId = 1, 
                                               outcomeIds = 3, 
                                               cohortDatabaseSchema = "main", 
                                               cohortTable = "cohort", 
                                               outcomeDatabaseSchema = "main", 
                                               outcomeTable =  "cohort", 
                                               firstExposureOnly = T, 
                                               washoutPeriod = 365, 
                                               covariateSettings = covSet
)

if(temp){
plpDataT <- PatientLevelPrediction::getPlpData(connectionDetails = connectionDetails, 
                                   cdmDatabaseSchema = "main",
                                   cohortId = 1, 
                                   outcomeIds = 3, 
                                   cohortDatabaseSchema = "main", 
                                   cohortTable = "cohort", 
                                   outcomeDatabaseSchema = "main", 
                                   outcomeTable =  "cohort", 
                                   firstExposureOnly = T, 
                                   washoutPeriod = 365, 
                                   covariateSettings = covSetT
                                   )
}


population <- PatientLevelPrediction::createStudyPopulation(plpData = plpData, 
                                                            outcomeId = 3, 
                                                            requireTimeAtRisk = F, 
                                                            riskWindowStart = 1, 
                                                            riskWindowEnd = 365)

##sparseMat <- toSparseRTorch(plpData, population, map=NULL, temporal=T)
if(F){
x <- toSparseMDeep(plpData ,population, 
              map=NULL, 
              temporal=F)

x2 <- toSparseMDeep(plpDataT ,population, 
                   map=NULL, 
                   temporal=T)
}

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


resSet <- setResNet(numLayers=5, sizeHidden=256, hiddenFactor=2,
                    residualDropout=c(0.1), 
                    hiddenDropout=c(0.1),
                    normalization='BatchNorm', activation= 'RelU',
                    sizeEmbedding=64, weightDecay=c(1e-6),
                    learningRate=c(3e-4), seed=42, hyperParamSearch='random',
                    randomSample=1, 
                    device='cuda:0', 
                    batchSize=128, 
                    epochs=10)


res2 <- runPlp(population = population, 
              plpData = plpData, 
              nfold = 3,
              modelSettings = resSet, 
              savePlpData = F, 
              savePlpResult = F, 
              savePlpPlots = F, 
              saveEvaluation = F)

##predict.customLibrary(libraryName, predictionFunction, inputList){
##  libraryName <- 'PatientLevelPrediction'
##  predictionFunction <- "createStudyPopulation"
##  predictFun <- get(predictionFunction, envir = rlang::search_envs()[grep(paste0('package:', libraryName), search())][[1]])
##  
##  prediction <- do.call(predictFun, inputList)
##  return(prediction)
##}
