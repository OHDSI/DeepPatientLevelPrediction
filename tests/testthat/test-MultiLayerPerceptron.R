
modelSettings <- setMultiLayerPerceptron(
  numLayers = 2,
  sizeHidden = 32,
  dropout = c(0.1),
  sizeEmbedding = 32,
  estimatorSettings = setEstimator(
    learningRate = c(3e-4),
    weightDecay = c(1e-6),
    seed = 42,
    batchSize = 128,
    epochs = 1
  ),
  hyperParamSearch = "random",
  randomSample = 1
)

test_that("setMultiLayerPerceptron works", {
  expect_s3_class(object = modelSettings, class = "modelSettings")

  expect_equal(modelSettings$fitFunction, "DeepPatientLevelPrediction::fitEstimator")

  expect_true(length(modelSettings$param) > 0)

  expect_error(setMultiLayerPerceptron(numLayers = 1,
                                       sizeHidden = 128,
                                       dropout = 0.2,
                                       sizeEmbedding = 128,
                                       estimatorSettings =
                                         setEstimator(learningRate = 3e-4),
                                       randomSample = 2))
})

sink(nullfile())
results <- tryCatch(
  {
    PatientLevelPrediction::runPlp(
      plpData = plpData,
      outcomeId = 3,
      modelSettings = modelSettings,
      analysisId = "Analysis_MLP",
      analysisName = "Testing Deep Learning",
      populationSettings = populationSet,
      splitSettings = PatientLevelPrediction::createDefaultSplitSetting(),
      sampleSettings = PatientLevelPrediction::createSampleSettings(),
      featureEngineeringSettings =
        PatientLevelPrediction::createFeatureEngineeringSettings(),
      preprocessSettings = PatientLevelPrediction::createPreprocessSettings(),
      executeSettings = PatientLevelPrediction::createExecuteSettings(
        runSplitData = TRUE,
        runSampleData = FALSE,
        runFeatureEngineering = FALSE,
        runPreprocessData = TRUE,
        runModelDevelopment = TRUE,
        runCovariateSummary = FALSE
      ),
      saveDirectory = file.path(testLoc, "MLP")
    )
  },
  error = function(e) {
    print(e)
    return(NULL)
  }
)
sink()

test_that("MLP with runPlp working checks", {
  expect_false(is.null(results))

  # check structure
  expect_true("prediction" %in% names(results))
  expect_true("model" %in% names(results))
  expect_true("covariateSummary" %in% names(results))
  expect_true("performanceEvaluation" %in% names(results))

  # check prediction same size as pop
  expect_equal(nrow(results$prediction %>%
                                dplyr::filter(evaluationType %in% c("Train",
                                                                    "Test"))),
                         nrow(population))

  # check prediction between 0 and 1
  expect_gte(min(results$prediction$value), 0)
  expect_lte(max(results$prediction$value), 1)
})


test_that("MLP nn-module works ", {
  mlp <- reticulate::import_from_path("MultiLayerPerceptron", path = path)$MultiLayerPerceptron
  model <- mlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 5,
    size_hidden = 16,
    num_layers = 1,
    activation = torch$nn$ReLU,
    normalization = torch$nn$BatchNorm1d,
    dropout = 0.3
  )

  pars <- sum(reticulate::iterate(model$parameters(), function(x) x$numel()))

  # expected number of parameters
  expect_equal(pars, 976)

  input <- list()
  input$feature_ids <- torch$randint(0L, 5L, c(10L, 5L), dtype = torch$long)
  input$feature_values <- torch$randint(0L, 5L, c(10L, 5L), dtype = torch$long)


  output <- model(input)

  # output is correct shape
  expect_equal(output$shape[0], 10L)

  model <- mlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 5L,
    size_hidden = 16L,
    num_layers = 1L,
    activation = torch$nn$ReLU,
    normalization = torch$nn$BatchNorm1d,
    dropout = 0.3,
    dim_out = 1L
  )
  output <- model(input)
  # model works without numeric variables
  expect_equal(output$shape[0], 10L)
})


test_that("Errors are produced by settings function", {
  randomSample <- 2

  expect_error(setMultiLayerPerceptron(
                                       numLayers = 1,
                                       sizeHidden = 128,
                                       dropout = 0.0,
                                       sizeEmbedding = 128,
                                       hyperParamSearch = "random",
                                       estimatorSettings =
                                         setEstimator(
                                                      learningRate = "auto",
                                                      weightDecay = c(1e-3),
                                                      batchSize = 1024,
                                                      epochs = 30,
                                                      device = "cpu")))
})

test_that("Can upload results to database", {
  cohortDefinitions <- data.frame(
    cohortName = c("blank1"),
    cohortId = c(1),
    json = c("json")
  )

  sink(nullfile())
  sqliteFile <-
    PatientLevelPrediction::insertResultsToSqlite(resultLocation = file.path(testLoc, "MLP"),
                          cohortDefinitions = cohortDefinitions)
  sink()

  testthat::expect_true(file.exists(sqliteFile))

  cdmDatabaseSchema <- "main"
  ohdsiDatabaseSchema <- "main"
  connectionDetails <- DatabaseConnector::createConnectionDetails(
    dbms = "sqlite",
    server = sqliteFile
  )
  conn <- DatabaseConnector::connect(connectionDetails = connectionDetails)
  targetDialect <- "sqlite"

  # check the results table is populated
  sql <- "select count(*) as N from main.performances;"
  res <- DatabaseConnector::querySql(conn, sql)
  expect_true(res$N[1] > 0)
})
