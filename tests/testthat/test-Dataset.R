test_that("dataset correct class", {
  testthat::expect_true("myDataset" %in% class(dataset))
})

test_that("length of index correct", {
  testthat::expect_equal(
    length(dataset$getNumericalIndex()),
    dplyr::n_distinct(mappedData$covariates %>% dplyr::pull(covariateId))
  )
})

test_that("number of num and cat features sum correctly", {
  testthat::expect_equal(
    dataset$numNumFeatures() + dataset$numCatFeatures(),
    dplyr::n_distinct(mappedData$covariates %>% dplyr::pull(covariateId))
  )
})


test_that("length of dataset correct", {
  expect_equal(length(dataset), dataset$cat$shape[1])
  expect_equal(length(dataset), dataset$num$shape[1])
  expect_equal(
    dataset$.length(),
    dplyr::n_distinct(mappedData$covariates %>% dplyr::pull(rowId))
  )
})

test_that(".getbatch works", {
  batch_size <- 16
  # get one sample
  out <- dataset[10]

  # output should be a list of two items, the batch and targets,
  # the batch is what goes to the model
  expect_equal(length(out), 2)

  # targets should be binary
  expect_true(out$target$item() %in% c(0, 1))

  # shape of batch is correct
  expect_equal(length(out$batch), 2)
  expect_equal(out$batch$cat$shape[1], 1)
  expect_equal(out$batch$num$shape[1], 1)

  # shape of target
  expect_equal(out$target$shape[1], 1)

  # get a whole batch
  out <- dataset[10:(10 + batch_size - 1)]

  expect_equal(length(out), 2)
  expect_true(all(torch::as_array(out$target) %in% c(0, 1)))

  expect_equal(length(out$batch), 2)
  expect_equal(out$batch$cat$shape[1], 16)
  expect_equal(out$batch$num$shape[1], 16)

  expect_equal(out$target$shape[1], 16)
})
