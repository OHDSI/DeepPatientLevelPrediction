test_that("number of num and cat features sum correctly", {
  skip_if_no_python()
  featureInfo <- simDataset$get_feature_info()
  testthat::expect_equal(
    featureInfo$get_vocabulary_size(),
    dplyr::n_distinct(simMappedData$covariates %>%
      dplyr::collect() %>%
      dplyr::pull(covariateId))
  )
})


test_that("length of simDataset correct", {
  skip_if_no_python()
  expect_equal(length(simDataset), simDataset$data[["feature_ids"]]$shape[[0]])
  expect_equal(length(simDataset), simDataset$data[["feature_values"]]$shape[[0]])
  expect_equal(
    length(simDataset),
    dplyr::n_distinct(simMappedData$covariates %>%
      dplyr::collect() %>%
      dplyr::pull(.data$rowId))
  )
})

test_that(".getbatch works", {
  skip_if_no_python()
  batchSize <- 16
  # get one sample
  out <- simDataset[10]

  # output should be a list of two items,
  # the batch in pos 1 and targets in pos 2,
  # the batch is what goes to the model
  expect_equal(length(out), 2)

  # targets should be binary
  expect_true(out[[2]]$item() %in% c(0, 1))

  # shape of batch is correct
  expect_equal(length(out[[1]]), 3)
  expect_equal(out[[1]]$feature_ids$shape[0], 1)
  expect_equal(out[[1]]$feature_values$shape[0], 1)

  # shape of target
  expect_equal(out[[2]]$shape$numel(), 1)

  # get a whole batch
  seq <- 10:(10 + batchSize - 1)
  out <- simDataset[seq]

  expect_equal(length(out), 2)
  expect_true(all(out[[2]]$numpy() %in% c(0, 1)))

  expect_equal(length(out[[1]]), 3)
  expect_equal(out[[1]]$feature_ids$shape[0], 16)
  expect_equal(out[[1]]$feature_values$shape[0], 16)

  expect_equal(out[[2]]$shape[0], 16)
})

test_that("Column order is preserved when features are missing", {
  skip_if_no_python()
  # important for transfer learning and external validation
  reducedCovData <- Andromeda::copyAndromeda(simTrainData$Train$covariateData)

  # remove one numerical and one categorical
  numFeature <- 1002 # continous age
  # the most common condition
  catFeature <- simTrainData$Train$covariateData$covariates %>%
    dplyr::filter(.data$covariateId %in% local(
      simTrainData$Train$covariateData$covariateRef %>%
        dplyr::filter(.data$analysisId == 102) %>%
        dplyr::pull("covariateId")
    )) %>%
    dplyr::collect() %>%
    dplyr::count(.data$covariateId, sort = TRUE) %>%
    utils::head(1) %>%
    dplyr::pull("covariateId")
  reducedCovData$covariates <- simTrainData$Train$covariateData$covariates %>%
    dplyr::filter(!(covariateId %in% c(numFeature, catFeature)))
  reducedCovData$covariates <- simTrainData$Train$covariateData$covariates %>%
    dplyr::filter(!(covariateId %in% c(numFeature, catFeature)))

  mappedReducedData <- PatientLevelPrediction::MapIds(
    reducedCovData,
    mapping = simMappedData$mapping
  )

  catColumn <- simMappedData$mapping %>%
    dplyr::filter(covariateId == catFeature) %>%
    dplyr::pull("columnId")
  numColumn <- simMappedData$mapping %>%
    dplyr::filter(covariateId == numFeature) %>%
    dplyr::pull("columnId")

  reducedDataset <- createDataset(
    data = mappedReducedData,
    labels = simTrainData$Train$labels,
  )

  # all zeros in columns with removed feature
  expect_true(reducedDataset$data[["feature_values"]][reducedDataset$data[["feature_ids"]] == numColumn]$sum()$item() == 0)

  # all other columns are same
  indexReduced <- !torch$isin(
    reducedDataset$data[["feature_ids"]],
    torch$as_tensor(c(numColumn, catColumn))
  )
  index <- !torch$isin(
    simDataset$data[["feature_ids"]],
    torch$as_tensor(c(numColumn, catColumn))
  )


  expect_equal(
    reducedDataset$data[["feature_values"]][indexReduced]$sum()$item(),
    simDataset$data[["feature_values"]][index]$sum()$item()
  )

  # not same counts for removed feature
  expect_false(isTRUE(all.equal(
    (reducedDataset$data[["feature_ids"]] == catColumn)$sum()$item(),
    (simDataset$data[["feature_ids"]] == catColumn)$sum()$item()
  )))

  # get counts
  counts <- as.array(torch$unique(simDataset$data[["feature_ids"]],
    return_counts = TRUE
  )[[2]]$numpy())
  counts <- counts[-(catColumn + 1)] # +1 because py_to_r
  counts <- counts[-1]

  reducedCounts <- as.array(torch$unique(reducedDataset$data[["feature_ids"]],
    return_counts = TRUE
  )[[2]]$numpy())
  reducedCounts <- reducedCounts[-(catColumn + 1)] # +1 because py_to_r
  reducedCounts <- reducedCounts[-1]

  expect_false(isTRUE(all.equal(counts, reducedCounts)))
  expect_equal(
    simDataset$get_feature_info()$get_vocabulary_size(),
    reducedDataset$get_feature_info()$get_vocabulary_size()
  )
})

test_that("feature_ids and feature_values remain in sync in dataset with timeIds", {
  skip_if_no_python()
  analysisRef <- data.frame(
    analysisId = 1,
    analysisName = "dummy",
    domainId = "Condition",
    startDay = -365,
    endDay = -1,
    isBinary = "N",
    missingMeansZero = "N"
  )
  covariateRef <- data.frame(
    covariateId = c(1, 2, 3, 4, 5, 6),
    columnId = c(1L, 2L, 3L, 4L, 5L, 6L),
    covariateName = paste0("Cov_", 1:6),
    analysisId = c(1, 1, 1, 1, 1, 1)
  )
  covariates <- data.frame(
    rowId = c(1L, 1L, 1L, 2L, 2L, 2L),
    covariateId = c(5, 2, 4, 6, 3, 1),
    covariateValue = c(0.5, 0.2, 0.4, 1.0, 2.0, 3.0),
    timeId = c(10, 10, 12, 7, 7, 8),
    columnId = c(5L, 2L, 4L, 6L, 3L, 1L)
  )
  timeRef <- data.frame(
    timePart = "month",
    timeInterval = 1,
    sequenceStartDay = -365,
    sequenceEndDay = -1
  )
  covData <- Andromeda::andromeda(
    analysisRef = analysisRef,
    covariateRef = covariateRef,
    covariates = covariates,
    timeRef = timeRef
  )
  dataset <- createDataset(
    data = covData,
    labels = data.frame(rowId = c(1L, 2L), outcomeCount = c(1L, 0L)),
    temporalSettings = list(
      useRope = FALSE,
      maxSequenceLength = 12L,
      truncation = "tail",
      timeTokens = FALSE
    )
  )

  featureIds <- dataset$data[["feature_ids"]]
  featureValues <- dataset$data[["feature_values"]]
  rowIds <- dataset$data[["row_ids"]]
  for (i in seq_along(rowIds)) {
    id <- rowIds[i - 1]$item()
    datasetFeatures <- data.frame(
      ids = featureIds[i - 1][0:3]$tolist(),
      values = featureValues[i - 1][0:3]$tolist()
    )
    datasetFeatures <- datasetFeatures %>% dplyr::arrange(.data$ids)

    expectedFeatures <- covariates %>% 
      dplyr::filter(.data$rowId == id + 1) %>% # +1 because py_to_r
      dplyr::arrange(.data$columnId) %>%
      dplyr::select("columnId", "covariateValue")

    expect_equal(datasetFeatures$ids, expectedFeatures$columnId)
    expect_equal(datasetFeatures$values, expectedFeatures$covariateValue)
  }
})

test_that("predictions are kept for patients without covariates", {
  skip_if_no_python()
  data <- createMockData(data.frame(
    rowId = c(1, 1, 3),
    covariateId = c(101, 102, 101),
    columnId = c(1L, 2L, 1L),
    covariateValue = c(1, 1, 2)
  ))
  withr::defer(Andromeda::close(data))
  plpModel <- list(covariateImportance = data.frame(
    columnId = 1:2, covariateId = c(101, 102), analysisId = 1,
    conceptId = c(1001, 1002), covariateName = c("a", "b")
  ))
  dataset <- createDataset(data, plpModel = plpModel, numRows = 4)

  expect_equal(reticulate::py_len(dataset), 4L)
  featureIds <- dataset$data$feature_ids$numpy()
  expect_equal(unname(featureIds[1, ]), c(1, 2))
  expect_equal(unname(featureIds[2, ]), c(0, 0))
  expect_equal(unname(featureIds[3, ]), c(1, 0))
  expect_equal(unname(featureIds[4, ]), c(0, 0))

  settingsList <- list(
    setMultiLayerPerceptron(
      numLayers = 1, sizeHidden = 4, dropout = 0, sizeEmbedding = 4,
      estimatorSettings = setEstimator(learningRate = 0.01, device = "cpu", seed = 42),
      randomSample = 1
    ),
    setResNet(
      numLayers = 1, sizeHidden = 4, hiddenFactor = 1, residualDropout = 0,
      hiddenDropout = 0, sizeEmbedding = 4,
      estimatorSettings = setEstimator(learningRate = 0.01, device = "cpu", seed = 42),
      randomSample = 1
    ),
    setTransformer(
      numBlocks = 1, dimToken = 4, dimOut = 1, numHeads = 2, attDropout = 0,
      ffnDropout = 0, dimHidden = 4,
      estimatorSettings = setEstimator(learningRate = 0.01, device = "cpu", seed = 42),
      randomSample = 1
    ),
    setRealMLP(sizeHidden = 4L, numLayers = 1L, sizeEmbedding = 4L)
  )
  for (settings in settingsList) {
    parameters <- settings$param[[1]]
    parameters$modelType <- settings$modelType
    parameters$feature_info <- dataset$get_feature_info()
    estimator <- createEstimator(list(
      modelParameters = parameters,
      estimatorSettings = settings$estimatorSettings
    ))
    predictions <- as.numeric(estimator$predict_proba(dataset))
    expect_length(predictions, 4)
    expect_true(all(is.finite(predictions)), info = settings$modelType)
  }
})

test_that("training keeps patients without covariates aligned with their labels", {
  skip_if_no_python()
  data <- createMockData(data.frame(
    rowId = c(1, 3), covariateId = c(101, 101), columnId = c(1L, 1L),
    covariateValue = c(1, 2)
  ))
  withr::defer(Andromeda::close(data))
  dataset <- createDataset(data, data.frame(rowId = 1:3, outcomeCount = c(1, 0, 1)))
  expect_equal(reticulate::py_len(dataset), 3L)
  expect_equal(as.numeric(dataset$data$feature_values$numpy()), c(1, 0, 2))
})

test_that("covariates outside of the cohort are rejected", {
  skip_if_no_python()
  data <- createMockData(data.frame(
    rowId = 5, covariateId = 101, columnId = 1L, covariateValue = 1
  ))
  withr::defer(Andromeda::close(data))
  expect_error(
    createDataset(data, data.frame(rowId = 1:2, outcomeCount = c(0, 1))),
    "only 2 patients"
  )
})

test_that("a mismatch between predictions and the cohort is an error", {
  estimator <- list(predict_proba = function(data) c(0.1, 0.9))
  expect_error(
    predictDeepEstimator(estimator, data = NULL, cohort = data.frame(rowId = 1:4)),
    "2 predictions for 4 patients"
  )
})

test_that("datasets join analysis IDs stored as integers and doubles", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  # a saved model's covariate reference comes back from CSV with integer IDs,
  # while the analysis reference of new data can hold doubles
  data$covariateRef <- data$covariateRef %>%
    dplyr::collect() %>%
    dplyr::mutate(analysisId = as.integer(.data$analysisId))
  data$analysisRef <- data$analysisRef %>%
    dplyr::collect() %>%
    dplyr::mutate(analysisId = as.double(.data$analysisId))
  dataset <- createDataset(data, createMockLabels())
  expect_equal(length(dataset), 8)
})
