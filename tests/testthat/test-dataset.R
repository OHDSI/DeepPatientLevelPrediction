context("dataset")


test_that("dataset correct class", {
testthat::expect_true("myDataset" %in%class(dataset))
})

test_that("length of index correct", {
  
testthat::expect_equal(
length(dataset$getNumericalIndex()),
dim(mappedData$dataMatrix)[2]
)

})

test_that("number of num and cat features sum correctly", {
  
testthat::expect_equal(
  dataset$numNumFeatures()+dataset$numCatFeatures(), 
  dim(mappedData$dataMatrix)[2]
)

})

test_that("dataset rows size match", {
  
testthat::expect_equal(
  dataset$.length(), 
  dim(mappedData$dataMatrix)[1]
)

})
