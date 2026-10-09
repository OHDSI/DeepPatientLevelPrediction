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
        unlist(estimator$model$embedding$custom_indices$tolist()), c(1L, 2L)
      )
      expect_true(torch$allclose(
        estimator$model$embedding$custom_embeddings$weight,
        torch$tensor(rbind(c(0, 0), weights[1:2, ]), dtype = torch$float)
      ))
    }
  }
})
