# Models saved by an earlier version of the package must keep loading and
# predicting the same. The fixtures are written by
# fixtures/make-saved-models.R. Do not regenerate them to make this test pass:
# a failure means that models users saved with that version would break.

savedModelVersions <- Filter(
  function(path) startsWith(basename(path), "saved-models-"),
  list.dirs(test_path("fixtures"), recursive = FALSE)
)

# loads a saved model from a copy, with its weights under their usual name
loadSavedModel <- function(modelDirectory) {
  copy <- file.path(withr::local_tempdir(.local_envir = parent.frame()), basename(modelDirectory))
  dir.create(copy)
  file.copy(list.files(modelDirectory, full.names = TRUE), copy, recursive = TRUE)
  file.rename(
    file.path(copy, "model", "DeepEstimatorModel.pt.bin"),
    file.path(copy, "model", "DeepEstimatorModel.pt")
  )
  PatientLevelPrediction::loadPlpModel(copy)
}

test_that("models saved by earlier versions predict the same", {
  skip_if_no_python()
  expect_true(length(savedModelVersions) > 0)
  for (version in savedModelVersions) {
    plpData <- PatientLevelPrediction::loadPlpData(file.path(version, "plpData"))
    population <- PatientLevelPrediction::createStudyPopulation(
      plpData, outcomeId = 3, populationSettings = simulatedPopulationSettings()
    )
    models <- setdiff(list.dirs(version, recursive = FALSE), file.path(version, "plpData"))
    for (modelDirectory in models) {
      label <- paste(basename(version), basename(modelDirectory))
      predicted <- quietly(PatientLevelPrediction::predictPlp(
        loadSavedModel(modelDirectory), plpData, population
      ))
      saved <- utils::read.csv(file.path(modelDirectory, "predictions.csv"))
      both <- merge(saved, predicted[, c("rowId", "value")], by = "rowId", suffixes = c("Saved", "Now"))
      expect_equal(nrow(both), nrow(saved), label = label)
      expect_equal(both$valueNow, both$valueSaved, tolerance = 1e-5, label = label)
    }
  }
})
