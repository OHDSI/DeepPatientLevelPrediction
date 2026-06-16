library(testthat)

run_tests <- function() {
  # Keep torch.compile/inductor artifacts inside this check temp dir and clean up.
  torchinductor_cache_dir <- file.path(
    tempdir(),
    sprintf("torchinductor_cache_%s", Sys.getpid())
  )
  Sys.setenv(TORCHINDUCTOR_CACHE_DIR = torchinductor_cache_dir)
  on.exit({
    unlink(torchinductor_cache_dir, recursive = TRUE, force = TRUE)
    Sys.unsetenv("TORCHINDUCTOR_CACHE_DIR")
  }, add = TRUE)

  library(DeepPatientLevelPrediction)

  filter <- Sys.getenv("DPLP_TEST_FILTER", unset = "")
  if (nzchar(filter)) {
    message("Running testthat filter: ", filter)
  }

  withCallingHandlers({
    if (nzchar(filter)) {
      test_check("DeepPatientLevelPrediction", filter = filter)
    } else {
      test_check("DeepPatientLevelPrediction")
    }
  }, error = function(e) {
    traceback()
    message(e)
    if (!is.null(reticulate::py_last_error())) {
      reticulate::py_last_error()
    }
    stop(e)
  })
}

run_tests()
