
modelSettings <- setMultiLayerPerceptron(
  numLayers = c(2),
  sizeHidden = c(32),
  dropout = c(0.1),
  sizeEmbedding = c(32),
  weightDecay = c(1e-6),
  learningRate = c(3e-4),
  seed = 42,
  hyperParamSearch = "random",
  randomSample = 1,
  batchSize = 128,
  epochs = 3
)

test_that("setMultiLayerPerceptron works", {
  testthat::expect_s3_class(object = modelSettings, class = "modelSettings")

  testthat::expect_equal(modelSettings$fitFunction, "fitEstimator")

  testthat::expect_true(length(modelSettings$param) > 0)
})

sink(nullfile())
results <- tryCatch(
  {
    PatientLevelPrediction::runPlp(
      plpData = plpData,
      outcomeId = 3,
      modelSettings = modelSettings,
      analysisId = "MLP",
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
  model <- MLP(
    catFeatures = 5, numFeatures = 1, sizeEmbedding = 5,
    sizeHidden = 16, numLayers = 1,
    activation = torch::nn_relu,
    normalization = torch::nn_batch_norm1d, dropout = 0.3,
    d_out = 1
  )

  pars <- sum(sapply(model$parameters, function(x) prod(x$shape)))

  # expected number of parameters
  expect_equal(pars, 489)

  input <- list()
  input$cat <- torch::torch_randint(0, 5, c(10, 5), dtype = torch::torch_long())
  input$num <- torch::torch_randn(10, 1, dtype = torch::torch_float32())


  output <- model(input)

  # output is correct shape
  expect_equal(output$shape, 10)

  input$num <- NULL
  model <- MLP(
    catFeatures = 5, numFeatures = 0, sizeEmbedding = 5,
    sizeHidden = 16, numLayers = 1,
    activation = torch::nn_relu,
    normalization = torch::nn_batch_norm1d, dropout = 0.3,
    d_out = 1
  )
  output <- model(input)
  # model works without numeric variables
  expect_equal(output$shape, 10)
})
