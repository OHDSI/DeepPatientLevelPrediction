# Runs the test suite with coverage.py measuring the Python backend in
# inst/python, and writes coverage-python.xml for upload. covr measures the R
# code separately; it cannot see Python called through reticulate.
Sys.setenv(DPLP_RUN_PYTHON_TESTS = "true")
# load the package before Python starts, so that it imports torch its own way
# (see #191)
pkgload::load_all(".", quiet = TRUE)
coverage <- reticulate::import("coverage")
measure <- coverage$Coverage(source = list(normalizePath("inst/python")), data_file = NULL)
measure$set_option("run:relative_files", TRUE)
measure$start()
results <- testthat::test_dir(
  "tests/testthat",
  package = "DeepPatientLevelPrediction",
  load_package = "none",
  reporter = testthat::SummaryReporter$new(),
  stop_on_failure = FALSE
)
measure$stop()
measure$xml_report(outfile = "coverage-python.xml")
cat(sprintf("Python coverage: %.1f%%\n", measure$report(file = reticulate::import("io")$StringIO())))
