test_that("custom embeddings match float and integer concept IDs", {
  skip_if_no_python()

  path <- system.file("python", package = "DeepPatientLevelPrediction")
  pl <- reticulate::import("polars", convert = FALSE)
  featureInfo <- reticulate::import_from_path(
    "Dataset", path = path, convert = FALSE
  )$FeatureInfo
  embeddingFile <- withr::local_tempfile(fileext = ".pt")
  weights <- matrix(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), ncol = 2, byrow = TRUE)
  torch$save(list(
    concept_ids = torch$tensor(c(1001L, 1002L, 9999L), dtype = torch$long),
    embeddings = torch$tensor(weights, dtype = torch$float)
  ), embeddingFile)

  for (conceptIds in list(c(0, 1001, 1002, 1003), c(0L, 1001L, 1002L, 1003L))) {
    dataReference <- pl$DataFrame(list(
      columnId = 1:4,
      conceptId = conceptIds,
      isBinary = rep("Y", 4)
    ))

    for (embeddingClass in c("CustomEmbeddings", "PoincareEmbeddings")) {
      settings <- setCustomEmbeddingModel(
        embeddingFilePath = embeddingFile,
        modelSettings = setResNet(
          numLayers = 1, sizeHidden = 4, hiddenFactor = 1,
          residualDropout = 0, hiddenDropout = 0, sizeEmbedding = 2,
          estimatorSettings = setEstimator(learningRate = 0.001, seed = 42),
          randomSample = 1
        ),
        embeddingsClass = embeddingClass
      )
      parameters <- settings$param[[1]]
      parameters$modelType <- settings$modelType
      parameters$feature_info <- featureInfo(dataReference)
      estimator <- createEstimator(list(
        modelParameters = parameters,
        estimatorSettings = settings$estimatorSettings
      ))

      expect_equal(
        unlist(estimator$model$embedding$custom_indices$tolist()), c(2L, 3L)
      )
      expect_true(torch$allclose(
        estimator$model$embedding$custom_embeddings$weight,
        torch$tensor(rbind(c(0, 0), weights[1:2, ]), dtype = torch$float)
      ))
    }
  }
})

customEmbeddingEstimator <- function(concepts, weights, conceptIds = 1:4 + 1000,
                                     isBinary = rep("Y", length(conceptIds))) {
  embeddingFile <- withr::local_tempfile(fileext = ".pt", .local_envir = parent.frame())
  torch$save(list(
    concept_ids = torch$tensor(as.array(as.integer(concepts)), dtype = torch$long),
    embeddings = torch$tensor(weights, dtype = torch$float)
  ), embeddingFile)
  settings <- setCustomEmbeddingModel(
    embeddingFilePath = embeddingFile,
    modelSettings = setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 4, dropout = 0, sizeEmbedding = 2,
      estimatorSettings = setEstimator(learningRate = 0.001, device = "cpu", seed = 42),
      randomSample = 1
    )
  )
  parameters <- settings$param[[1]]
  parameters$modelType <- settings$modelType
  parameters$feature_info <- mappedFeatureInfo(conceptIds = conceptIds, isBinary = isBinary)
  createEstimator(list(
    modelParameters = parameters,
    estimatorSettings = settings$estimatorSettings
  ))
}

# the patient's embedding: its tokens, pooled as the model pools them
embedRow <- function(estimator, ids) {
  featureIds <- torch$tensor(matrix(as.integer(ids), nrow = 1), dtype = torch$long)
  tokens <- estimator$model$embedding(list(
    feature_ids = featureIds,
    feature_values = torch$ones(featureIds$shape, dtype = torch$float)
  ))
  asNumeric(estimator$model$`_aggregate_tokens`(tokens, featureIds))
}

test_that("custom embeddings are assigned to their own covariates", {
  skip_if_no_python()
  # concepts in the file are in the opposite order of their columns
  estimator <- customEmbeddingEstimator(
    concepts = c(1003, 1001),
    weights = rbind(c(3, 30), c(1, 10))
  )
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), c(3L, 1L))
  expect_equal(embedRow(estimator, 1), c(1, 10))
  expect_equal(embedRow(estimator, 3), c(3, 30))
})

test_that("custom embeddings work for a single matching concept", {
  skip_if_no_python()
  estimator <- customEmbeddingEstimator(concepts = 1002, weights = rbind(c(5, 50)))
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), 2L)
  expect_equal(embedRow(estimator, 2), c(5, 50))
})

test_that("custom embedding covariates sharing a concept each get its vector", {
  skip_if_no_python()
  estimator <- customEmbeddingEstimator(
    concepts = 1001,
    weights = rbind(c(7, 70)),
    conceptIds = c(1001, 1002, 1001, 1003)
  )
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), c(1L, 3L))
  expect_equal(embedRow(estimator, 1), c(7, 70))
  expect_equal(embedRow(estimator, 3), c(7, 70))
})

test_that("a concept with several vectors in the embeddings file is an error", {
  skip_if_no_python()
  expect_error(
    customEmbeddingEstimator(concepts = c(1002, 1003, 1002), weights = rbind(c(1, 2), c(3, 4), c(5, 6))),
    "more than one vector for concept IDs 1002"
  )
  # a repeat that is not in the training data is dropped like other unused concepts
  estimator <- customEmbeddingEstimator(
    concepts = c(1002, 9999, 9999), weights = rbind(c(1, 2), c(3, 4), c(5, 6))
  )
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), 2L)
})

test_that("aggregated custom embeddings do not depend on padding", {
  skip_if_no_python()
  estimator <- customEmbeddingEstimator(concepts = 1001, weights = rbind(c(1, 10)))
  # one custom covariate with one regular covariate
  unpadded <- embedRow(estimator, c(1, 2))
  expect_equal(embedRow(estimator, c(1, 2, 0, 0)), unpadded, tolerance = 1e-6)
  regular <- embedRow(estimator, 2)
  expect_equal(unpadded, (c(1, 10) + regular) / 2, tolerance = 1e-6)
  expect_equal(embedRow(estimator, c(1, 0, 0)), c(1, 10), tolerance = 1e-6)
})

test_that("the projection of custom embeddings only applies to custom features", {
  skip_if_no_python()
  # two custom embeddings with two dimensions, projected to three
  vectors <- rbind(c(1, 2), c(3, 4))
  customEmbedding <- function() {
    importModule("CustomEmbeddings")$CustomEmbeddings(
      custom_embedding_weights = torch$tensor(vectors, dtype = torch$float),
      embedding_dim = 3L,
      feature_info = mappedFeatureInfo(),
      custom_indices = torch$tensor(c(1L, 3L), dtype = torch$long)
    )
  }
  position <- function(tokens, i) asNumeric(tokens$select(1L, i)$reshape(-1L))
  project <- function(embedding, row) {
    asNumeric(embedding$linear_transform(torch$tensor(vectors[row, , drop = FALSE], dtype = torch$float)))
  }
  torch$manual_seed(42L)

  perToken <- customEmbedding()
  regular <- embedTokens(perToken$regular_embeddings, c(0, 2, 0))
  tokens <- embedTokens(perToken, c(1, 2, 0))
  expect_equal(position(tokens, 0L), project(perToken, 1), tolerance = 1e-6)
  expect_equal(position(tokens, 1L), position(regular, 1L), tolerance = 1e-6)
  expect_equal(position(tokens, 2L), c(0, 0, 0))
})

test_that("custom embeddings leave the batch they are given unchanged", {
  skip_if_no_python()
  estimator <- customEmbeddingEstimator(concepts = 1001, weights = rbind(c(1, 10)))
  ids <- torch$tensor(matrix(c(1L, 2L), nrow = 1), dtype = torch$long)
  batch <- reticulate::dict(
    feature_ids = ids,
    feature_values = torch$ones(c(1L, 2L), dtype = torch$float)
  )
  first <- asNumeric(estimator$model$embedding(batch))
  expect_equal(asNumeric(reticulate::py_get_item(batch, "feature_ids")), c(1, 2))
  expect_equal(asNumeric(reticulate::py_get_item(batch, "feature_values")), c(1, 1))
  expect_equal(asNumeric(estimator$model$embedding(batch)), first)
})

test_that("custom embeddings leave numeric covariates to their value embedding", {
  skip_if_no_python()
  # column 1 is numeric, such as a measurement value, and column 3 binary;
  # both have a concept in the embeddings file
  estimator <- customEmbeddingEstimator(
    concepts = c(1001, 1003),
    weights = rbind(c(1, 10), c(3, 30)),
    isBinary = c("N", "Y", "Y", "Y")
  )
  expect_equal(unlist(estimator$model$embedding$custom_indices$tolist()), 3L)
  embedValue <- function(value) {
    asNumeric(estimator$model$embedding(list(
      feature_ids = torch$tensor(matrix(1L, nrow = 1), dtype = torch$long),
      feature_values = torch$tensor(matrix(value, nrow = 1), dtype = torch$float)
    )))
  }
  expect_false(isTRUE(all.equal(embedValue(1), embedValue(7.5))))
  expect_equal(embedRow(estimator, 3), c(3, 30))
})
