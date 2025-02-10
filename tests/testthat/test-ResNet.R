
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

  expect_error(setResNet(numLayers = 2,
                         sizeHidden = 32,
                         hiddenFactor = 2,
                         residualDropout = 0.1,
                         hiddenDropout = 0.1,
                         sizeEmbedding = 32,
                         estimatorSettings =
                           setEstimator(learningRate = c(3e-4),
                                        weightDecay = c(1e-6),
                                        seed = 42,
                                        batchSize = 128,
                                        epochs = 1),
                         hyperParamSearch = "random",
                         randomSample = 2))
})

sink(nullfile())
res2 <- tryCatch(
  {
    PatientLevelPrediction::runPlp(
      plpData = plpData,
      outcomeId = 3,
      modelSettings = resSet,
      analysisId = "Analysis_ResNet",
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
      saveDirectory = file.path(testLoc, "ResNet")
    )
  },
  error = function(e) {
    print(e)
    return(NULL)
  }
)
sink()

test_that("ResNet with runPlp working checks", {
  expect_false(is.null(res2))

  # check structure
  expect_true("prediction" %in% names(res2))
  expect_true("model" %in% names(res2))
  expect_true("covariateSummary" %in% names(res2))
  expect_true("performanceEvaluation" %in% names(res2))

  # check prediction same size as pop
  expect_equal(nrow(res2$prediction %>%
                                dplyr::filter(evaluationType %in% c("Train",
                                                                    "Test"))),
                         nrow(population))

  # check prediction between 0 and 1
  expect_gte(min(res2$prediction$value), 0)
  expect_lte(max(res2$prediction$value), 1)
})


test_that("ResNet nn-module works ", {
  resNet <- reticulate::import_from_path("ResNet", path = path)$ResNet
  model <- resNet(
    feature_info = list("categorical_features" = 5L,
                    "numerical_features" = 1L),
    size_embedding = 5,
    size_hidden = 16,
    num_layers = 1,
    hidden_factor = 2,
    activation = torch$nn$ReLU,
    normalization = torch$nn$BatchNorm1d,
    hidden_dropout = 0.3,
    residual_dropout = 0.3
  )

  pars <- sum(reticulate::iterate(model$parameters(), function(x) x$numel()))

  # expected number of parameters
  expect_equal(pars, 1295)

  input <- list()
  input$cat <- torch$randint(0L, 5L, c(10L, 5L), dtype = torch$long)
  input$num <- torch$randn(10L, 1L, dtype = torch$float32)


  output <- model(input)

  # output is correct shape
  expect_equal(output$shape[0], 10L)

  input$num <- NULL
  model <- resNet(
    feature_info = list("categorical_features" = 5L,
                    "numerical_features" = 0),
    size_embedding = 5,
    size_hidden = 16,
    num_layers = 1,
    hidden_factor = 2,
    activation = torch$nn$ReLU,
    normalization = torch$nn$BatchNorm1d,
    hidden_dropout = 0.3,
    residual_dropout = 0.3
  )
  output <- model(input)
  # model works without numeric variables
  expect_equal(output$shape[0], 10L)
})

test_that("Default Resnet works", {
  defaultResNet <- setDefaultResNet()
  params <- defaultResNet$param[[1]]

  expect_equal(params$numLayers, 6)
  expect_equal(params$sizeHidden, 512)
  expect_equal(params$hiddenFactor, 2)
  expect_equal(params$residualDropout, 0.1)
  expect_equal(params$hiddenDropout, 0.4)
  expect_equal(params$sizeEmbedding, 256)

})

test_that("Errors are produced by settings function", {
  randomSample <- 2

  expect_error(setResNet(numLayers = 1,
                         sizeHidden = 128,
                         hiddenFactor = 1,
                         residualDropout = 0.0,
                         hiddenDropout = 0.0,
                         sizeEmbedding = 128,
                         estimatorSettings = setEstimator(weightDecay = 1e-6,
                                                          learningRate = 0.01,
                                                          seed = 42),
                         hyperParamSearch = "random",
                         randomSample = randomSample))
})


test_that("Can upload results to database", {
  cohortDefinitions <- data.frame(
    cohortName = c("blank1"),
    cohortId = c(1),
    json = c("json")
  )

  sink(nullfile())
  sqliteFile <-
    insertResultsToSqlite(resultLocation = file.path(testLoc, "ResNet"),
                          cohortDefinitions = cohortDefinitions)
  sink()

  expect_true(file.exists(sqliteFile))

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
  testthat::expect_true(res$N[1] > 0)
})
