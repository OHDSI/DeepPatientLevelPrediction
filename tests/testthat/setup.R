# Python tests and the Eunomia integration tests are opted into separately, so
# that the Python tests run offline and without the slow integration setup.
.pythonRequested <- identical(Sys.getenv("DPLP_RUN_PYTHON_TESTS"), "true")
.integrationRequested <- identical(Sys.getenv("DPLP_RUN_INTEGRATION_TESTS"), "true")
.integrationPackages <- c("curl", "DatabaseConnector", "Eunomia", "FeatureExtraction")
.missingIntegrationPackages <- .integrationPackages[
  !vapply(.integrationPackages, requireNamespace, logical(1), quietly = TRUE)
]
.runIntegrationTests <- .pythonRequested && .integrationRequested &&
  length(.missingIntegrationPackages) == 0 && curl::has_internet()

skip_if_no_python <- function() {
  testthat::skip_if_not(
    .pythonRequested,
    "Set DPLP_RUN_PYTHON_TESTS=true to run Python tests"
  )
}

skip_if_no_integration <- function() {
  reason <- if (!.pythonRequested) {
    "Set DPLP_RUN_PYTHON_TESTS=true to run Python tests"
  } else if (!.integrationRequested) {
    "Set DPLP_RUN_INTEGRATION_TESTS=true to run the Eunomia integration tests"
  } else if (length(.missingIntegrationPackages) > 0) {
    paste(
      "Integration tests require:",
      paste(.missingIntegrationPackages, collapse = ", ")
    )
  } else {
    "Eunomia integration tests require internet access"
  }
  testthat::skip_if_not(.runIntegrationTests, reason)
}

# Eunomia data for the tests that need a real database: the runPlp() checks
# and result uploads of ResNet and MLP, and the temporal Transformer. Tests
# that only need data use the simulated fixtures in helper-fixtures.R.
if (.runIntegrationTests) {
connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

covSet <- FeatureExtraction::createCovariateSettings(
  useDemographicsGender = TRUE,
  useDemographicsAge = TRUE,
  useDemographicsRace = TRUE,
  useDemographicsEthnicity = TRUE,
  useDemographicsAgeGroup = TRUE,
  useConditionGroupEraLongTerm = TRUE,
  useDrugEraStartLongTerm = TRUE,
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

restrictPlpDataSettings <-
  PatientLevelPrediction::createRestrictPlpDataSettings(
    firstExposureOnly = TRUE,
    washoutPeriod = 365
  )

plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  restrictPlpDataSettings = restrictPlpDataSettings,
  covariateSettings = covSet
)

populationSet <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = FALSE,
  riskWindowStart = 1,
  riskWindowEnd = 365
)

population <- PatientLevelPrediction::createStudyPopulation(
  plpData = plpData,
  outcomeId = 3,
  populationSettings = populationSet
)
}
