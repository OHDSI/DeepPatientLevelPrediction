test_that("integration setup is opt-in and does not probe the network by default", {
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "false")
  guardEnv <- new.env()
  guardEnv$requireNamespace <- function(...) TRUE
  sys.source(test_path("setup.R"), envir = guardEnv)

  expect_false(guardEnv$.runIntegrationTests)
  expect_condition(guardEnv$skip_if_no_integration(), class = "skip")
  expect_condition(guardEnv$skip_if_no_python(), class = "skip")
})

test_that("missing integration packages skip setup before downloading data", {
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "true")

  for (missingPackage in c("curl", "DatabaseConnector", "Eunomia", "FeatureExtraction")) {
    guardEnv <- new.env()
    guardEnv$requireNamespace <- function(package, ...) package != missingPackage
    sys.source(test_path("setup.R"), envir = guardEnv)

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
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "true")
  local_mocked_bindings(has_internet = function(...) FALSE, .package = "curl")
  guardEnv <- new.env()
  guardEnv$requireNamespace <- function(...) TRUE
  sys.source(test_path("setup.R"), envir = guardEnv)

  expect_false(guardEnv$.runIntegrationTests)
  expect_condition(
    guardEnv$skip_if_no_integration(),
    regexp = "internet access",
    class = "skip"
  )
  expect_no_condition(guardEnv$skip_if_no_python())
  expect_false(exists("connectionDetails", envir = guardEnv, inherits = FALSE))
})

test_that("online opt-in reaches integration setup when dependencies are present", {
  skip_if_not_installed("curl")
  skip_if_not_installed("Eunomia")
  withr::local_envvar(DPLP_RUN_PYTHON_TESTS = "true")
  local_mocked_bindings(has_internet = function(...) TRUE, .package = "curl")
  local_mocked_bindings(
    getEunomiaConnectionDetails = function(...) stop("integration setup reached"),
    .package = "Eunomia"
  )
  guardEnv <- new.env()
  guardEnv$requireNamespace <- function(...) TRUE

  expect_error(
    sys.source(test_path("setup.R"), envir = guardEnv),
    "integration setup reached"
  )
  expect_true(guardEnv$.runIntegrationTests)
})
