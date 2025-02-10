library(testthat)
library(DeepPatientLevelPrediction)

withCallingHandlers({
  test_check("DeepPatientLevelPrediction")
}, error = function(e) {
  traceback()
  message(e)
  if (!is.null(reticulate::py_last_error())) {
    message(reticulate::py_last_error())
  }
  stop(e)
})
