sourceSetup <- function(envir) {
  sys.source(test_path("setup.R"), envir = envir)
}

test_that("integration setup is opt-in and does not probe the network by default", {
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "false", DPLP_RUN_INTEGRATION_TESTS = NA)
  guardEnv <- new.env()
  guardEnv$requireNamespace <- function(...) TRUE
  sourceSetup(guardEnv)

  expect_false(guardEnv$.runIntegrationTests)
  expect_condition(guardEnv$skip_if_no_integration(), class = "skip")
  expect_condition(guardEnv$skip_if_no_python(), class = "skip")
})

test_that("Python tests run without the integration setup unless it is requested", {
  skip_if_not_installed("curl")
  skip_if_not_installed("Eunomia")
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "true", DPLP_RUN_INTEGRATION_TESTS = NA)
  local_mocked_bindings(has_internet = function(...) TRUE, .package = "curl")
  local_mocked_bindings(
    getEunomiaConnectionDetails = function(...) stop("integration setup reached"),
    .package = "Eunomia"
  )
  guardEnv <- new.env()
  guardEnv$requireNamespace <- function(...) TRUE
  sourceSetup(guardEnv)

  expect_false(guardEnv$.runIntegrationTests)
  expect_no_condition(guardEnv$skip_if_no_python())
  expect_condition(
    guardEnv$skip_if_no_integration(),
    regexp = "DPLP_RUN_INTEGRATION_TESTS",
    class = "skip"
  )
})

test_that("missing integration packages skip setup before downloading data", {
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "true", DPLP_RUN_INTEGRATION_TESTS = "true")

  for (missingPackage in c("curl", "DatabaseConnector", "Eunomia", "FeatureExtraction")) {
    guardEnv <- new.env()
    guardEnv$requireNamespace <- function(package, ...) package != missingPackage
    sourceSetup(guardEnv)

    expect_identical(guardEnv$.missingIntegrationPackages, missingPackage)
    expect_false(guardEnv$.runIntegrationTests)
    expect_condition(
      guardEnv$skip_if_no_integration(),
      regexp = missingPackage,
      class = "skip"
    )
    # Local Python tests do not need these R packages or the Eunomia data.
    expect_no_condition(guardEnv$skip_if_no_python())
  }
})

test_that("offline integration setup skips before accessing Eunomia or Python", {
  skip_if_not_installed("curl")
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "true", DPLP_RUN_INTEGRATION_TESTS = "true")
  local_mocked_bindings(has_internet = function(...) FALSE, .package = "curl")
  guardEnv <- new.env()
  guardEnv$requireNamespace <- function(...) TRUE
  sourceSetup(guardEnv)

  expect_false(guardEnv$.runIntegrationTests)
  expect_condition(
    guardEnv$skip_if_no_integration(),
    regexp = "internet access",
    class = "skip"
  )
  expect_no_condition(guardEnv$skip_if_no_python())
  expect_false(exists("connectionDetails", envir = guardEnv, inherits = FALSE))
})

test_that("both opt-ins reach integration setup when dependencies are present", {
  skip_if_not_installed("curl")
  skip_if_not_installed("Eunomia")
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "true", DPLP_RUN_INTEGRATION_TESTS = "true")
  local_mocked_bindings(has_internet = function(...) TRUE, .package = "curl")
  local_mocked_bindings(
    getEunomiaConnectionDetails = function(...) stop("integration setup reached"),
    .package = "Eunomia"
  )
  guardEnv <- new.env()
  guardEnv$requireNamespace <- function(...) TRUE

  expect_error(sourceSetup(guardEnv), "integration setup reached")
  expect_true(guardEnv$.runIntegrationTests)
})
