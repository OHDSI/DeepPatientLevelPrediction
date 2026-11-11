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
