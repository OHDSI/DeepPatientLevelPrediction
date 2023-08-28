
modelSettings <- setMultiLayerPerceptron(
  numLayers = 2L,
  sizeHidden = 32L,
  dropout = c(0.1),
  sizeEmbedding = 32L,
  estimatorSettings = setEstimator(
    learningRate=c(3e-4),
    weightDecay = c(1e-6),
    seed=42,
    batchSize=128L,
    epochs=1L
  ),
  hyperParamSearch = "random",
  randomSample = 1
)

test_that("setMultiLayerPerceptron works", {
  testthat::expect_s3_class(object = modelSettings, class = "modelSettings")

  testthat::expect_equal(modelSettings$fitFunction, "fitEstimator")

  testthat::expect_true(length(modelSettings$param) > 0)
  
  expect_error(setMultiLayerPerceptron(numLayers=1,
                                       sizeHidden = 128,
                                       dropout= 0.2,
                                       sizeEmbedding = 128,
                                       estimatorSettings = setEstimator(learningRate=3e-4),
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
      sampleSettings = PatientLevelPrediction::createSampleSettings(), # none
      featureEngineeringSettings = PatientLevelPrediction::createFeatureEngineeringSettings(), # none
      preprocessSettings = PatientLevelPrediction::createPreprocessSettings(),
      executeSettings = PatientLevelPrediction::createExecuteSettings(
        runSplitData = T,
        runSampleData = F,
        runfeatureEngineering = F,
        runPreprocessData = T,
        runModelDevelopment = T,
        runCovariateSummary = F
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
  testthat::expect_false(is.null(results))

  # check structure
  testthat::expect_true("prediction" %in% names(results))
  testthat::expect_true("model" %in% names(results))
  testthat::expect_true("covariateSummary" %in% names(results))
  testthat::expect_true("performanceEvaluation" %in% names(results))

  # check prediction same size as pop
  testthat::expect_equal(nrow(results$prediction %>%
    dplyr::filter(evaluationType %in% c("Train", "Test"))), nrow(population))

  # check prediction between 0 and 1
  testthat::expect_gte(min(results$prediction$value), 0)
  testthat::expect_lte(max(results$prediction$value), 1)
})


test_that("MLP nn-module works ", {
  MLP <- reticulate::import_from_path("MLP", path=path)$MLP
  model <- MLP(
    cat_features = 5L, 
    num_features = 1L, 
    size_embedding = 5L,
    size_hidden = 16L, 
    num_layers = 1L,
    activation = torch$nn$ReLU,
    normalization = torch$nn$BatchNorm1d, 
    dropout = 0.3
  )

  pars <- sum(reticulate::iterate(model$parameters(), function(x) x$numel()))

  # expected number of parameters
  expect_equal(pars, 489)

  input <- list()
  input$cat <- torch$randint(0L, 5L, c(10L, 5L), dtype=torch$long)
  input$num <- torch$randn(10L, 1L, dtype=torch$float32)


  output <- model(input)

  # output is correct shape
  expect_equal(output$shape[0], 10L)

  input$num <- NULL
  model <- MLP(
    cat_features = 5L, 
    num_features = 0, 
    size_embedding = 5L,
    size_hidden = 16L, 
    num_layers = 1L,
    activation = torch$nn$ReLU,
    normalization = torch$nn$BatchNorm1d, 
    dropout = 0.3,
    d_out = 1L
  )
  output <- model(input)
  # model works without numeric variables
  expect_equal(output$shape[0], 10L)
})


test_that("Errors are produced by settings function", {
  randomSample <- 2
  
  expect_error(setMultiLayerPerceptron(
    numLayers = 1L,
    sizeHidden = 128L,
    dropout = 0.0,
    sizeEmbedding = 128L,
    hyperParamSearch = 'random',
    estimatorSettings = setEstimator(
      learningRate = 'auto',
      weightDecay = c(1e-3),
      batchSize = 1024L,
      epochs = 30L,
      device="cpu")))
                           
})

test_that("Can upload results to database", {
  cohortDefinitions = data.frame(
    cohortName = c('blank1'), 
    cohortId = c(1), 
    json = c('json')
  )
  
  sink(nullfile())
  sqliteFile <- insertResultsToSqlite(resultLocation = file.path(testLoc, "MLP"),
                           cohortDefinitions = cohortDefinitions)
  sink()
  
  testthat::expect_true(file.exists(sqliteFile))
  
  cdmDatabaseSchema <- 'main'
  ohdsiDatabaseSchema <- 'main'
  connectionDetails <- DatabaseConnector::createConnectionDetails(
    dbms = 'sqlite',
    server = sqliteFile
  )
  conn <- DatabaseConnector::connect(connectionDetails = connectionDetails)
  targetDialect <- 'sqlite'
  
  # check the results table is populated
  sql <- 'select count(*) as N from main.performances;'
  res <- DatabaseConnector::querySql(conn, sql)
  testthat::expect_true(res$N[1]>0)
  
  
})