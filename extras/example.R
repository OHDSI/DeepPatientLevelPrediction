# testing code (requires sequential branch of FeatureExtraction):
library(FeatureExtraction)
library(DeepPatientLevelPrediction)
connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

covSet <- createTemporalSequenceCovariateSettings(useDemographicsGender = T, 
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

population <- PatientLevelPrediction::createStudyPopulation(plpData = plpData, 
                                                            outcomeId = 3, 
                                                            requireTimeAtRisk = F, 
                                                            riskWindowStart = 1, 
                                                            riskWindowEnd = 365)

sparseMat <- toSparseRTorch(plpData, population, map=NULL, temporal=T)

# code to train models

  

