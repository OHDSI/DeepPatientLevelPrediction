fineTunerSettings <- setFinetuner(
  modelPath = file.path(fitEstimatorPath, "plpModel"),
  estimatorSettings = setEstimator(device = "cpu",
                                   learningRate = 3e-4,
                                   batchSize = 128,
                                   epochs = 1)
)

test_that("Finetuner settings work", {
  expect_equal(fineTunerSettings$param[[1]]$modelPath,
               file.path(fitEstimatorPath, "plpModel"))
  expect_equal(fineTunerSettings$estimatorSettings$device, "cpu")
  expect_equal(fineTunerSettings$estimatorSettings$batchSize, 128)
  expect_equal(fineTunerSettings$estimatorSettings$epochs, 1)  
  expect_equal(fineTunerSettings$fitFunction, "fitEstimator")
  expect_equal(fineTunerSettings$saveType, "file")
  expect_equal(fineTunerSettings$modelType, "Finetuner")
  expect_equal(fineTunerSettings$modelParamNames, "modelPath")
  expect_equal(class(fineTunerSettings), "modelSettings")
  expect_equal(attr(fineTunerSettings$param, "settings")$modelType, "Finetuner")
  expect_error(setFinetuner(modelPath = "notAPath", estimatorSettings = setEstimator()))
  expect_error(setFinetuner(modelPath = fitEstimatorPath, estimatorSettings = setEstimator()))
  fakeDir <- file.path(fitEstimatorPath, "fakeDir")
  fakeSubdir <- file.path(fakeDir, "model")
  dir.create(fakeSubdir, recursive = TRUE)
  expect_error(setFinetuner(modelPath = fakeDir, estimatorSettings = setEstimator()))
  })

test_that("Finetuner fitEstimator works", {
  fineTunerPath <- file.path(testLoc, "fineTuner")
  dir.create(fineTunerPath)
  fineTunerResults <- fitEstimator(trainData$Train,
                                   modelSettings = fineTunerSettings,
                                   analysisId = 1,
                                   analysisPath = fineTunerPath)
  expect_equal(which(fineTunerResults$covariateImportance$isNumeric), 
               which(fitEstimatorResults$covariateImportance$isNumeric))
  expect_equal(nrow(fineTunerResults$covariateImportance),
               nrow(fitEstimatorResults$covariateImportance))
  expect_equal(fineTunerResults$covariateImportance$columnId,
               fitEstimatorResults$covariateImportance$columnId)
  expect_equal(fineTunerResults$covariateImportance$covariateId,
               fitEstimatorResults$covariateImportance$covariateId)
  
  fineTunedModel <- torch$load(file.path(fineTunerResults$model,
                                         "DeepEstimatorModel.pt"))
  expect_true(fineTunedModel$estimator_settings$finetune)
  expect_equal(fineTunedModel$estimator_settings$finetune_model_path, 
               file.path(fitEstimatorPath, "plpModel", "model",
                         "DeepEstimatorModel.pt"))   
  expect_equal(fineTunedModel$model_parameters$model_type, 
               fitEstimatorResults$modelDesign$modelSettings$modelType)
})