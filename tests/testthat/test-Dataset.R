test_that("number of num and cat features sum correctly", {
  testthat::expect_equal(
    length(dataset$get_numerical_features()) + length(dataset$get_cat_features()),
    dplyr::n_distinct(mappedData$covariates %>% dplyr::collect() %>%
                        dplyr::pull(covariateId))
  )
})


test_that("length of dataset correct", {
  expect_equal(length(dataset), dataset$cat$shape[0])
  expect_equal(length(dataset), dataset$num$shape[0])
  expect_equal(
    length(dataset),
    dplyr::n_distinct(mappedData$covariates %>% 
                        dplyr::collect() %>% dplyr::pull(rowId))
  )
})

test_that(".getbatch works", {
  batch_size <- 16
  # get one sample
  out <- dataset[10]
  
  # output should be a list of two items, the batch in pos 1 and targets in pos 2,
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
  out <- dataset[10:(10 + batch_size - 1)]
  
  expect_equal(length(out), 2)
  expect_true(all(out[[2]]$numpy() %in% c(0, 1)))
  
  expect_equal(length(out[[1]]), 2)
  expect_equal(out[[1]]$cat$shape[0], 16)
  expect_equal(out[[1]]$num$shape[0], 16)
  
  expect_equal(out[[2]]$shape[0], 16)
})
