library(PatientLevelPrediction)

testLoc <- tempdir()
torch <- reticulate::import("torch")
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

mappedData <- PatientLevelPrediction::MapIds(
  covariateData = trainData$Train$covariateData,
  cohort = trainData$Train$labels
)

path <- system.file("python", package = "DeepPatientLevelPrediction")
Dataset <- reticulate::import_from_path("Dataset", path = path)
if (is.null(attributes(mappedData)$path)) {
  # sqlite object
  attributes(mappedData)$path <- attributes(mappedData)$dbname
}

dataset <- Dataset$Data(
  data = reticulate::r_to_py(attributes(mappedData)$path),
  labels = reticulate::r_to_py(trainData$Train$labels$outcomeCount),
)
small_dataset <- torch$utils$data$Subset(dataset, (1:round(length(dataset)/3)))

