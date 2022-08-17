
resSet <- setResNet(
  numLayers = c(2),
  sizeHidden = c(32),
  hiddenFactor = c(2),
  residualDropout = c(0.1),
  hiddenDropout = c(0.1),
  sizeEmbedding = c(32),
  weightDecay = c(1e-6),
  learningRate = c(3e-4),
  seed = 42,
  hyperParamSearch = "random",
  randomSample = 1,
  # device='cuda:0',
  batchSize = 128,
  epochs = 3
)

test_that("setResNet works", {
  testthat::expect_s3_class(object = resSet, class = "modelSettings")

  testthat::expect_equal(resSet$fitFunction, "fitEstimator")

  testthat::expect_true(length(resSet$param) > 0)
})

sink(nullfile())
res2 <- tryCatch(
  {
    PatientLevelPrediction::runPlp(
      plpData = plpData,
      outcomeId = 3,
      modelSettings = resSet,
      analysisId = "ResNet",
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
      saveDirectory = file.path(testLoc, "Deep")
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
  model <- ResNet(
    catFeatures = 5, numFeatures = 1, sizeEmbedding = 5,
    sizeHidden = 16, numLayers = 1, hiddenFactor = 2,
    activation = torch::nn_relu,
    normalization = torch::nn_batch_norm1d, hiddenDropout = 0.3,
    residualDropout = 0.3, d_out = 1
  )

  pars <- sum(sapply(model$parameters, function(x) prod(x$shape)))

  # expected number of parameters
  expect_equal(pars, 1295)

  input <- list()
  input$cat <- torch::torch_randint(0, 5, c(10, 5), dtype = torch::torch_long())
  input$num <- torch::torch_randn(10, 1, dtype = torch::torch_float32())


  output <- model(input)

  # output is correct shape
  expect_equal(output$shape, 10)

  input$num <- NULL
  model <- ResNet(
    catFeatures = 5, numFeatures = 0, sizeEmbedding = 5,
    sizeHidden = 16, numLayers = 1, hiddenFactor = 2,
    activation = torch::nn_relu,
    normalization = torch::nn_batch_norm1d, hiddenDropout = 0.3,
    residualDropout = 0.3, d_out = 1
  )
  output <- model(input)
  # model works without numeric variables
  expect_equal(output$shape, 10)
})
