
resSet <- setResNet(
  numLayers = 2L,
  sizeHidden = 32L,
  hiddenFactor = 2L,
  residualDropout = 0.1,
  hiddenDropout = 0.1,
  sizeEmbedding = 32L,
  estimatorSettings = setEstimator(learningRate="auto",
                                   weightDecay = c(1e-6),
                                   seed=42,
                                   batchSize = 128L,
                                   epochs=1L),
  hyperParamSearch = "random",
  randomSample = 1,
)

test_that("setResNet works", {
  testthat::expect_s3_class(object = resSet, class = "modelSettings")
  
  testthat::expect_equal(resSet$fitFunction, "fitEstimator")
  
  testthat::expect_true(length(resSet$param) > 0)
  
  expect_error(setResNet(numLayers = 2L,
                         sizeHidden = 32L,
                         hiddenFactor = 2L,
                         residualDropout = 0.1,
                         hiddenDropout = 0.1,
                         sizeEmbedding = 32L,
                         estimatorSettings = setEstimator(learningRate=c(3e-4),
                                                          weightDecay = c(1e-6),
                                                          seed=42,
                                                          batchSize = 128L,
                                                          epochs=1L),
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
  testthat::expect_false(is.null(res2))
  
  # check structure
  testthat::expect_true("prediction" %in% names(res2))
  testthat::expect_true("model" %in% names(res2))
  testthat::expect_true("covariateSummary" %in% names(res2))
  testthat::expect_true("performanceEvaluation" %in% names(res2))
  
  # check prediction same size as pop
  testthat::expect_equal(nrow(res2$prediction %>%
                                dplyr::filter(evaluationType %in% c("Train", "Test"))), nrow(population))
  
  # check prediction between 0 and 1
  testthat::expect_gte(min(res2$prediction$value), 0)
  testthat::expect_lte(max(res2$prediction$value), 1)
})


test_that("ResNet nn-module works ", {
  ResNet <- reticulate::import_from_path("ResNet", path=path)$ResNet
  model <- ResNet(
    cat_features = 5L, 
    num_features = 1L, 
    size_embedding = 5L,
    size_hidden = 16L, 
    num_layers = 1L, 
    hidden_factor = 2L,
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
  model <- ResNet(
    cat_features = 5L, 
    num_features = 0L, 
    size_embedding = 5L,
    size_hidden = 16L, 
    num_layers = 1L, 
    hidden_factor = 2L,
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
  
  expect_equal(params$numLayers, 6L)
  expect_equal(params$sizeHidden, 512L)
  expect_equal(params$hiddenFactor, 2L)
  expect_equal(params$residualDropout, 0.1)
  expect_equal(params$hiddenDropout, 0.4)
  expect_equal(params$sizeEmbedding, 256L)
  
}) 

test_that("Errors are produced by settings function", {
  randomSample <- 2
  
  expect_error(setResNet(
    numLayers = 1L,
    sizeHidden = 128L,
    hiddenFactor = 1,
    residualDropout = 0.0,
    hiddenDropout = 0.0,
    sizeEmbedding = 128L,
    estimatorSettings = setEstimator(weightDecay = 1e-6,
                                     learningRate = 0.01,
                                     seed = 42),
    hyperParamSearch = 'random',
    randomSample = randomSample))
})


test_that("Can upload results to database", { 
  cohortDefinitions = data.frame(
    cohortName = c('blank1'), 
    cohortId = c(1), 
    json = c('json')
  )
  
  sink(nullfile())
  sqliteFile <- insertResultsToSqlite(resultLocation = file.path(testLoc, "ResNet"),
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

