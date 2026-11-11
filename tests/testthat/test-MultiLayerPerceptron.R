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

test_that("MLP nn-module works ", {
  skip_if_no_python()
  mlp <- importModule("MultiLayerPerceptron")$MultiLayerPerceptron
  newModel <- function(featureInfo) {
    mlp(
      feature_info = featureInfo,
      size_embedding = 5L,
      size_hidden = 16L,
      num_layers = 1L,
      activation = torch$nn$ReLU,
      normalization = torch$nn$BatchNorm1d,
      dropout = 0.3,
      dim_out = 1L
    )
  }
  # one numeric and four binary features
  model <- newModel(mappedFeatureInfo(
    covariateIds = 1:5 + 100, conceptIds = 1:5 + 1000,
    isBinary = c("N", "Y", "Y", "Y", "Y")
  ))

  pars <- sum(reticulate::iterate(model$parameters(), function(x) x$numel()))

  # expected number of parameters
  expect_equal(pars, 494)

  input <- list()
  input$feature_ids <- torch$randint(0L, 5L, c(10L, 5L), dtype = torch$long)
  input$feature_values <- torch$randint(0L, 5L, c(10L, 5L), dtype = torch$long)

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
