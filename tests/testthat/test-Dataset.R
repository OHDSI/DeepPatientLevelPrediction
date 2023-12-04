test_that("number of num and cat features sum correctly", {
  testthat::expect_equal(
    length(dataset$get_numerical_features()) +
      length(dataset$get_cat_features()),
    dplyr::n_distinct(mappedData$covariates %>%
                        dplyr::collect() %>%
                        dplyr::pull(covariateId))
  )
})


test_that("length of dataset correct", {
  expect_equal(length(dataset), dataset$cat$shape[0])
  expect_equal(length(dataset), dataset$num$shape[0])
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
  expect_equal(length(out[[1]]), 2)
  expect_equal(out[[1]]$cat$shape[0], 1)
  expect_equal(out[[1]]$num$shape[0], 1)

  # shape of target
  expect_equal(out[[2]]$shape$numel(), 1)

  # get a whole batch
  out <- dataset[10:(10 + batchSize)]

  expect_equal(length(out), 2)
  expect_true(all(out[[2]]$numpy() %in% c(0, 1)))

  expect_equal(length(out[[1]]), 2)
  expect_equal(out[[1]]$cat$shape[0], 16)
  expect_equal(out[[1]]$num$shape[0], 16)

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

  reducedDataset <- datasetClass$Data(
    data =
      reticulate::r_to_py(normalizePath(attributes(mappedReducedData)$dbname)),
    labels = reticulate::r_to_py(trainData$Train$labels$outcomeCount),
    numerical_features = dataset$numerical_features$to_list()
  )

  # should have same number of columns
  expect_equal(dataset$num$shape[[1]], reducedDataset$num$shape[[1]])

  # all zeros in column with removed feature, -1 because r to py
  expect_true(reducedDataset$num[, numColumn - 1]$sum()$item() == 0)

  # all other columns are same
  indexReduced <- !torch$isin(torch$arange(reducedDataset$num$shape[[1]]),
                              numColumn - 1)
  index <- !torch$isin(torch$arange(dataset$num$shape[[1]]),
                       numColumn - 1)

  expect_equal(reducedDataset$num[, indexReduced]$mean()$item(),
               dataset$num[, index]$mean()$item())

  # cat data should have same counts of all columnIds
  # expect the one that was removed
  # not same counts for removed feature
  expect_false(isTRUE(all.equal((reducedDataset$cat == catColumn)$sum()$item(),
                                (dataset$cat == catColumn)$sum()$item())))

  # get counts
  counts <- as.array(torch$unique(dataset$cat,
                                  return_counts = TRUE)[[2]]$numpy())
  counts <- counts[-(catColumn + 1)] # +1 because py_to_r
  counts <- counts[-1]

  reducedCounts <- as.array(torch$unique(reducedDataset$cat,
                                         return_counts = TRUE)[[2]]$numpy())
  reducedCounts <- reducedCounts[-(catColumn + 1)] # +1 because py_to_r
  reducedCounts <- reducedCounts[-1]

  expect_false(isTRUE(all.equal(counts, reducedCounts)))
  expect_equal(dataset$get_cat_features()$max(),
               reducedDataset$get_cat_features()$max())

})
