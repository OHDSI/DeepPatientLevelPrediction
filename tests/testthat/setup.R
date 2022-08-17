library(PatientLevelPrediction)

testLoc <- tempdir()

# get connection and data from Eunomia
connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

covSet <- FeatureExtraction::createCovariateSettings(
  useDemographicsGender = T,
  useDemographicsAge = T,
  useDemographicsRace = T,
  useDemographicsEthnicity = T,
  useDemographicsAgeGroup = T,
  useConditionGroupEraLongTerm = T,
  useDrugEraStartLongTerm = T,
  endDays = -1
)


covSetT <- FeatureExtraction::createTemporalSequenceCovariateSettings(
  useDemographicsGender = T,
  useDemographicsAge = T,
  useDemographicsRace = T,
  useDemographicsEthnicity = T,
  useDemographicsAgeGroup = T,
  useConditionEraGroupStart = T,
  useDrugEraStart = T,
  timePart = "month",
  timeInterval = 1,
  sequenceEndDay = -1,
  sequenceStartDay = -365 * 5
)


databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = "main",
  cohortDatabaseSchema = "main",
  cohortTable = "cohort",
  targetId = 4,
  outcomeIds = 3,
  outcomeDatabaseSchema = "main",
  outcomeTable = "cohort",
  cdmDatabaseName = "eunomia"
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


plpDataT <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  restrictPlpDataSettings = restrictPlpDataSettings,
  covariateSettings = covSetT
)


populationSet <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = F,
  riskWindowStart = 1,
  riskWindowEnd = 365
)

population <- PatientLevelPrediction::createStudyPopulation(
  plpData = plpData,
  outcomeId = 3,
  populationSettings = populationSet
)

trainData <- PatientLevelPrediction::splitData(
  plpData,
  population = population,
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting()
)

mappedData <- PatientLevelPrediction:::MapIds(
  covariateData = trainData$Train$covariateData,
  cohort = trainData$Train$labels
)

dataset <- Dataset(
  data = mappedData$covariates,
  labels = trainData$Train$labels$outcomeCount,
  numericalIndex = NULL
)
