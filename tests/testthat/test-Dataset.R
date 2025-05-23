test_that("number of num and cat features sum correctly", {
  featureInfo <- dataset$get_feature_info()
  testthat::expect_equal(
    featureInfo$get_vocabulary_size(),
    dplyr::n_distinct(mappedData$covariates %>%
      dplyr::collect() %>%
      dplyr::pull(covariateId))
  )
})


test_that("length of dataset correct", {
  expect_equal(length(dataset), dataset$data[["feature_ids"]]$shape[[0]])
  expect_equal(length(dataset), dataset$data[["feature_values"]]$shape[[0]])
  expect_equal(
    length(dataset),
    dplyr::n_distinct(mappedData$covariates %>%
      dplyr::collect() %>%
      dplyr::pull(.data$rowId))
  )
})

test_that(".getbatch works", {
  batchSize <- 16
  # get one sample
  out <- dataset[10]

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
  out <- dataset[10:(10 + batchSize)]

  expect_equal(length(out), 2)
  expect_true(all(out[[2]]$numpy() %in% c(0, 1)))

  expect_equal(length(out[[1]]), 3)
  expect_equal(out[[1]]$feature_ids$shape[0], 16)
  expect_equal(out[[1]]$feature_values$shape[0], 16)

  expect_equal(out[[2]]$shape[0], 16)
})

test_that("Column order is preserved when features are missing", {
  # important for transfer learning and external validation
  reducedCovData <- Andromeda::copyAndromeda(trainData$Train$covariateData)

  # remove one numerical and one categorical
  numFeature <- 1002 # continous age
  catFeature <- 4285898210 # a random common cat feature
  reducedCovData$covariates <- trainData$Train$covariateData$covariates %>%
    dplyr::filter(!(covariateId %in% c(numFeature, catFeature)))
  reducedCovData$covariates <- trainData$Train$covariateData$covariates %>%
    dplyr::filter(!(covariateId %in% c(numFeature, catFeature)))

  mappedReducedData <- PatientLevelPrediction::MapIds(
    reducedCovData,
    mapping = mappedData$mapping
  )

  catColumn <- mappedData$mapping %>%
    dplyr::filter(covariateId == catFeature) %>%
    dplyr::pull("columnId")
  numColumn <- mappedData$mapping %>%
    dplyr::filter(covariateId == numFeature) %>%
    dplyr::pull("columnId")

  reducedDataset <- createDataset(
    data = mappedReducedData,
    labels = trainData$Train$labels,
  )

  # all zeros in columns with removed feature
  expect_true(reducedDataset$data[["feature_values"]][reducedDataset$data[["feature_ids"]] == numColumn]$sum()$item() == 0)

  # all other columns are same
  indexReduced <- !torch$isin(reducedDataset$data[["feature_ids"]],
                              torch$as_tensor(c(numColumn, catColumn)))
  index <- !torch$isin(dataset$data[["feature_ids"]],
                      torch$as_tensor(c(numColumn, catColumn)))


  expect_equal(
    reducedDataset$data[["feature_values"]][indexReduced]$sum()$item(),
    dataset$data[["feature_values"]][index]$sum()$item()
  )

  # not same counts for removed feature
  expect_false(isTRUE(all.equal(
    (reducedDataset$data[["feature_ids"]] == catColumn)$sum()$item(),
    (dataset$data[["feature_ids"]] == catColumn)$sum()$item()
  )))

  # get counts
  counts <- as.array(torch$unique(dataset$data[["feature_ids"]],
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
    dataset$get_feature_info()$get_vocabulary_size(),
    reducedDataset$get_feature_info()$get_vocabulary_size()
  )
})

