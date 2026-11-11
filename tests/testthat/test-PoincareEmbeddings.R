# Models with Poincare embeddings keep the supplied vectors for the concepts
# in the data. They are fitted on the simulated training data; their full PLP
# round trip is covered by test-SaveLoadPredict.R.

# points of the Poincare ball for the condition concepts of the simulated data:
# each coordinate is below 0.5 in absolute value, so every norm is below
# sqrt(3) / 2
delayedAssign("poincareEmbeddings", local({
  conceptIds <- simPlpData$covariateData$covariateRef %>%
    dplyr::filter(.data$analysisId == 102) %>%
    dplyr::pull("conceptId") %>%
    unique()
  torch$manual_seed(42L)
  embeddings <- list(
    concept_ids = torch$tensor(conceptIds, dtype = torch$long),
    embeddings = torch$rand(length(conceptIds), 3L) - 0.5
  )
  file <- tempfile(fileext = ".pt")
  torch$save(embeddings, file)
  list(file = file, embeddings = embeddings)
}))

# Fits the model with Poincare embeddings and checks that the saved model holds
# the supplied vectors of the concepts in the data.
expectPoincareEmbeddingsKept <- function(modelSettings) {
  settings <- setCustomEmbeddingModel(
    embeddingFilePath = poincareEmbeddings$file,
    modelSettings = modelSettings,
    embeddingsClass = "PoincareEmbeddings"
  )
  results <- quietly(fitEstimator(
    simTrainData$Train,
    modelSettings = settings,
    analysisId = "Poincare",
    analysisPath = withr::local_tempdir()
  ))
  model <- torch$load(file.path(results$model, "DeepEstimatorModel.pt"), weights_only = FALSE)

  customModelEmbeddings <- model$model_state_dict$embedding.custom_embeddings.weight
  conceptsInData <- results$covariateImportance %>%
    dplyr::filter(.data$analysisId == 102) %>%
    dplyr::pull("conceptId")
  supplied <- poincareEmbeddings$embeddings
  embeddingsMask <- torch$isin(supplied[["concept_ids"]], torch$tensor(conceptsInData, dtype = torch$long))
  embeddingsInData <- supplied[["embeddings"]][embeddingsMask]

  expect_equal(customModelEmbeddings$shape[0] - 1, length(conceptsInData)) # -1 for padding embedding
  expect_equal(
    embeddingsInData$mean()$item(),
    customModelEmbeddings[1:customModelEmbeddings$shape[0], ]$mean()$item()
  )
}

test_that("Poincare Transformer works", {
  skip_if_no_python()
  expectPoincareEmbeddingsKept(setTransformer(
    numBlocks = 1,
    dimToken = 6,
    numHeads = 1,
    dimHidden = 12,
    estimatorSettings = quickEstimator()
  ))
})

test_that("ResNet with poincare works", {
  skip_if_no_python()
  expectPoincareEmbeddingsKept(setResNet(
    numLayers = 1,
    sizeEmbedding = 6,
    sizeHidden = 12,
    hiddenFactor = 1,
    residualDropout = 0.0,
    hiddenDropout = 0.0,
    estimatorSettings = quickEstimator(),
    randomSample = 1
  ))
})

test_that("MLP with poincare works", {
  skip_if_no_python()
  expectPoincareEmbeddingsKept(setMultiLayerPerceptron(
    numLayers = 1,
    sizeEmbedding = 6,
    sizeHidden = 12,
    dropout = 0.0,
    estimatorSettings = quickEstimator(),
    randomSample = 1
  ))
})

poincareEmbedding <- function(points, ...) {
  importModule("CustomEmbeddings")$PoincareEmbeddings(
    custom_embedding_weights = torch$tensor(points, dtype = torch$float),
    embedding_dim = ncol(points),
    feature_info = mappedFeatureInfo(),
    custom_indices = torch$tensor(array(seq_len(nrow(points))), dtype = torch$long),
    ...
  )
}

test_that("Poincare embeddings add the trainable offset in the tangent space", {
  skip_if_no_python()
  point <- c(0.3, 0.4)
  offset <- c(0.2, -0.1)
  embedding <- poincareEmbedding(rbind(point))
  embedding$custom_embeddings_trainable$weight$data$copy_(
    torch$tensor(rbind(c(0, 0), offset), dtype = torch$float)
  )
  # the log map at the origin of a point x with curvature 1
  tangent <- atanh(sqrt(sum(point^2))) / sqrt(sum(point^2)) * point
  tokens <- embedTokens(embedding, c(1, 2))
  expect_equal(asNumeric(tokens$select(1L, 0L)$reshape(-1L)), tangent + offset, tolerance = 1e-6)
})

test_that("Poincare embeddings must lie inside the ball of their curvature", {
  skip_if_no_python()
  # a norm of 0.6 is inside the ball with curvature -1 but not with -4
  point <- rbind(c(0.36, 0.48))
  expect_no_error(poincareEmbedding(point))
  expect_error(poincareEmbedding(point, curvature = 4), "not inside the Poincare ball")
  expect_error(poincareEmbedding(rbind(c(0.6, 0.8))), "not inside the Poincare ball")
  expect_error(poincareEmbedding(rbind(c(NaN, 0.1))), "not finite")
  expect_error(poincareEmbedding(point, curvature = 0), "positive number")
  expect_error(poincareEmbedding(point, freeze = FALSE), "needs frozen embeddings")
  # inside the ball, but sqrt(3) times its norm rounds to 1 in float32
  expect_error(poincareEmbedding(rbind(0.57735026), curvature = 3), "too close to the boundary")
  expect_no_error(poincareEmbedding(rbind(1 - 2^-24)))
})

test_that("the Poincare log map uses the curvature and is finite at the origin", {
  skip_if_no_python()
  logmap0 <- importModule("CustomEmbeddings")$logmap0
  point <- c(0.3, 0.4)
  for (curvature in c(1, 2)) {
    radius <- sqrt(curvature) * sqrt(sum(point^2))
    mapped <- logmap0(torch$tensor(rbind(point), dtype = torch$float), curvature)
    expect_equal(asNumeric(mapped), atanh(radius) / radius * point, tolerance = 1e-6)
  }
  origin <- torch$zeros(c(1L, 2L), dtype = torch$float, requires_grad = TRUE)
  mapped <- logmap0(origin, 1)
  mapped$sum()$backward()
  expect_equal(asNumeric(mapped), c(0, 0))
  expect_equal(asNumeric(origin$grad), c(1, 1))
})

test_that("the Poincare log map is accurate near the boundary and for small curvatures", {
  skip_if_no_python()
  logmap0 <- importModule("CustomEmbeddings")$logmap0
  mapOne <- function(x, curvature, dtype) {
    point <- torch$tensor(matrix(x), dtype = dtype, requires_grad = TRUE)
    mapped <- logmap0(point, curvature)
    mapped$sum()$backward()
    c(value = mapped$item(), gradient = point$grad$item())
  }
  # the largest float32 below 1 is inside the ball with curvature -1
  x <- 1 - 2^-24
  nearBoundary <- mapOne(x, 1, torch$float32)
  expect_equal(nearBoundary[["value"]], atanh(x), tolerance = 1e-5)
  expect_true(is.finite(nearBoundary[["gradient"]]))
  # 500 squared overflows float16, but with curvature -1e-6 the point is inside
  smallCurvature <- mapOne(500, 1e-6, torch$float16)
  expect_equal(smallCurvature[["value"]], atanh(0.5) / 0.5 * 500, tolerance = 1e-3)
  expect_true(is.finite(smallCurvature[["gradient"]]))
  # either side of where the series takes over
  for (x in c(0.0099, 0.0101)) {
    expect_equal(mapOne(x, 1, torch$float64)[["value"]], atanh(x), tolerance = 1e-12)
  }
})

test_that("setCustomEmbeddingModel passes the curvature to Poincare embeddings", {
  skip_if_no_python()
  embeddingFile <- withr::local_tempfile(fileext = ".pt")
  torch$save(list(
    concept_ids = torch$tensor(array(1001L), dtype = torch$long),
    embeddings = torch$tensor(rbind(c(0.3, 0.4)), dtype = torch$float)
  ), embeddingFile)
  settings <- setCustomEmbeddingModel(
    embeddingFilePath = embeddingFile,
    modelSettings = setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 4, dropout = 0, sizeEmbedding = 2,
      estimatorSettings = setEstimator(learningRate = 0.001, device = "cpu", seed = 42),
      randomSample = 1
    ),
    embeddingsClass = "PoincareEmbeddings",
    curvature = 2
  )
  parameters <- settings$param[[1]]
  parameters$modelType <- settings$modelType
  parameters$feature_info <- mappedFeatureInfo()
  estimator <- createEstimator(list(
    modelParameters = parameters,
    estimatorSettings = settings$estimatorSettings
  ))
  expect_equal(estimator$model$embedding$curvature, 2)
  expect_error(
    setCustomEmbeddingModel(embeddingFilePath = embeddingFile, curvature = 2),
    "only applies to"
  )
  for (curvature in list(-1, NA_real_, c(1, 2))) {
    expect_error(
      setCustomEmbeddingModel(
        embeddingFilePath = embeddingFile, embeddingsClass = "PoincareEmbeddings",
        curvature = curvature
      ),
      "single positive number"
    )
  }

  # an init strategy saved with a model before curvature was a setting
  strategy <- settings$estimatorSettings$initStrategy
  reticulate::py_del_attr(strategy, "curvature")
  estimator <- createEstimator(list(
    modelParameters = parameters,
    estimatorSettings = settings$estimatorSettings
  ))
  expect_equal(estimator$model$embedding$curvature, 1)
})
