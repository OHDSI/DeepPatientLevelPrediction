resSet <- setResNet(
  numLayers = 2,
  sizeHidden = 32,
  hiddenFactor = 2,
  residualDropout = 0.1,
  hiddenDropout = 0.1,
  sizeEmbedding = 32,
  estimatorSettings = setEstimator(learningRate = "auto",
                                   weightDecay = c(1e-6),
                                   seed = 42,
                                   batchSize = 128,
                                   epochs = 1),
  hyperParamSearch = "random",
  randomSample = 1,
)

test_that("setResNet works", {
  expect_s3_class(object = resSet, class = "modelSettings")

  expect_equal(resSet$fitFunction, "DeepPatientLevelPrediction::fitEstimator")

  expect_true(length(resSet$param) > 0)
})

# fitted on Eunomia when an integration test first uses it
delayedAssign("res2", quietly(PatientLevelPrediction::runPlp(
  plpData = plpData,
  outcomeId = 3,
  modelSettings = resSet,
  analysisId = "Analysis_ResNet",
  analysisName = "Testing Deep Learning",
  populationSettings = populationSet,
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting(),
  executeSettings = PatientLevelPrediction::createExecuteSettings(
    runSplitData = TRUE,
    runPreprocessData = TRUE,
    runModelDevelopment = TRUE
  ),
  saveDirectory = eunomiaResultsDirectory
)))

delayedAssign("eunomiaResultsDirectory", tempfile("ResNet"))

# A smoke test that a DPLP model runs end to end on real data. What runPlp()
# itself produces is PatientLevelPrediction's to test.
test_that("ResNet runs end to end with runPlp on Eunomia", {
  skip_if_no_integration()
  # runPlp() catches fitting errors and returns a placeholder model
  expect_equal(
    attr(res2$model, "predictionFunction"),
    "DeepPatientLevelPrediction::predictDeepEstimator"
  )
  trainAndTest <- res2$prediction %>%
    dplyr::filter(.data$evaluationType %in% c("Train", "Test"))
  expect_equal(nrow(trainAndTest), nrow(population))
  expect_gte(min(res2$prediction$value), 0)
  expect_lte(max(res2$prediction$value), 1)
})

test_that("ResNet nn-module works ", {
  skip_if_no_python()
  resNet <- importModule("ResNet")$ResNet
  newModel <- function(featureInfo) {
    resNet(
      feature_info = featureInfo,
      size_embedding = 5,
      size_hidden = 16,
      num_layers = 1,
      hidden_factor = 2,
      activation = torch$nn$ReLU,
      normalization = torch$nn$BatchNorm1d,
      hidden_dropout = 0.3,
      residual_dropout = 0.3
    )
  }
  # one numeric and four binary features
  model <- newModel(mappedFeatureInfo(
    covariateIds = 1:5 + 100, conceptIds = 1:5 + 1000,
    isBinary = c("N", "Y", "Y", "Y", "Y")
  ))

  pars <- sum(reticulate::iterate(model$parameters(), function(x) x$numel()))

  # expected number of parameters
  expect_equal(pars, 1294)

  input <- list()
  input[["feature_ids"]] <- torch$randint(1L, 5L, c(10L, 5L), dtype = torch$long)
  input[["feature_values"]] <- torch$randn(10L, 5L, dtype = torch$float32)

  output <- model(input)

  # output is correct shape
  expect_equal(output$shape[0], 10L)

  model <- newModel(mappedFeatureInfo(
    covariateIds = 1:5 + 100, conceptIds = 1:5 + 1000, isBinary = rep("Y", 5)
  ))
  output <- model(input)
  # model works without numeric variables
  expect_equal(output$shape[0], 10L)
})

test_that("a residual layer works without dropout", {
  skip_if_no_python()
  layer <- importModule("ResNet")$ResLayer(
    size_hidden = 4L, res_hidden = 8L, normalization = torch$nn$BatchNorm1d,
    activation = torch$nn$ReLU, hidden_dropout = NULL, residual_dropout = NULL
  )
  expect_equal(layer(torch$randn(3L, 4L))$shape[1], 4L)
})

test_that("setResNet() defaults to one model, which setDefaultResNet() gave", {
  resnet <- setResNet()
  expect_length(resnet$param, 1)
  params <- resnet$param[[1]]

  expect_equal(params$numLayers, 6)
  expect_equal(params$sizeHidden, 512)
  expect_equal(params$hiddenFactor, 2)
  expect_equal(params$residualDropout, 0.1)
  expect_equal(params$hiddenDropout, 0.4)
  expect_equal(params$sizeEmbedding, 256)
  expect_equal(resnet$estimatorSettings$weightDecay, 1e-6)
  expect_equal(resnet$estimatorSettings$epochs, 50)

  expect_warning(
    defaultResNet <- setDefaultResNet(),
    "`setDefaultResNet()` is deprecated",
    fixed = TRUE
  )
  expect_equal(defaultResNet$param, resnet$param)
  expect_equal(attr(defaultResNet, "settings")$name, "defaultResnet")
})

test_that("a random search with more samples than combinations runs each once", {
  settings <- setResNet(
    numLayers = 1:2, hyperParamSearch = "random", randomSample = 100,
    randomSampleSeed = 1
  )
  expect_setequal(vapply(settings$param, function(x) x$numLayers, numeric(1)), 1:2)
})


# PatientLevelPrediction stores the results; DPLP's part is that its model
# settings can be stored and read back.
test_that("DPLP model settings are stored with the results", {
  skip_if_not_installed("DatabaseConnector")
  skip_if_not_installed("ResultModelManager")
  skip_if_not_installed("RSQLite")
  skip_if_no_integration()
  # writes the results that are uploaded
  force(res2)

  sqliteFile <- quietly(PatientLevelPrediction::insertResultsToSqlite(
    resultLocation = eunomiaResultsDirectory,
    cohortDefinitions = data.frame(cohortName = "blank1", cohortId = 1, json = "json")
  ))
  connection <- DatabaseConnector::connect(DatabaseConnector::createConnectionDetails(
    dbms = "sqlite",
    server = sqliteFile
  ))
  withr::defer(DatabaseConnector::disconnect(connection))

  stored <- DatabaseConnector::querySql(
    connection,
    "SELECT model_type, model_settings_json FROM main.model_settings;",
    snakeCaseToCamelCase = TRUE
  )
  expect_equal(stored$modelType, "ResNet")
  expect_match(stored$modelSettingsJson, "DeepPatientLevelPrediction::fitEstimator", fixed = TRUE)
  expect_match(stored$modelSettingsJson, "numLayers", fixed = TRUE)
})
