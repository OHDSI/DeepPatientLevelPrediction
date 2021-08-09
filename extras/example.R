# testing code (requires sequential branch of FeatureExtraction):
rm(list = ls())
library(FeatureExtraction)
library(DeepPatientLevelPrediction)
connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

covSet <- createCovariateSettings(useDemographicsGender = T, 
                                                   useDemographicsAge = T, 
                                                   useDemographicsRace = T,
                                                   useDemographicsEthnicity = T, 
                                                   useDemographicsAgeGroup = T,
                                                   useConditionGroupEraLongTerm = T, 
                                                   useDrugEraStartLongTerm  = T, 
                                  endDays = -1
                                                   )

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

population <- PatientLevelPrediction::createStudyPopulation(plpData = plpData, 
                                                            outcomeId = 3, 
                                                            requireTimeAtRisk = F, 
                                                            riskWindowStart = 1, 
                                                            riskWindowEnd = 365)

##sparseMat <- toSparseRTorch(plpData, population, map=NULL, temporal=T)
x <- toSparseMDeep(plpData ,population, 
              map=NULL, 
              temporal=F)

x2 <- toSparseMDeep(plpDataT ,population, 
                   map=NULL, 
                   temporal=T)

# code to train models
deepset <- setDeepNNTorch(units=list(c(128, 64), 128), layer_dropout=c(0.2),
                          lr =c(1e-4), decay=c(1e-5), outcome_weight = c(1.0), batch_size = c(100), 
                          epochs= c(1),  seed=NULL  )

library(PatientLevelPrediction)

#debug(fitDeepNNTorch)
res <- runPlp(population = population, 
              plpData = plpData, 
              nfold = 3,
              modelSettings = deepset, 
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
