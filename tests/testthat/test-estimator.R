context("estimator")

matrixData <- mappedData$dataMatrix
labels <- mappedData$labels
dataset <- Dataset(matrixData, labels$outcomeCount)

test_that("estimator initialization works", {
  
  fitParams <- list()
  baseModel <- MRCovNN_submodel1 #(kernel_size = 4)
  
  estimator <- Estimator$new(
    baseModel = baseModel,
    modelParameters = list(kernel_size = 4),
    fitParameters = fitParams, 
    device = 'cpu'
  )
  testthat::expect_equal(
    estimator$model, 
    do.call(baseModel, list(kernel_size = 4))
    )
  
  testthat::expect_equal(
    estimator$modelParameters,
    list(kernel_size = 4)
  )
  testthat::expect_equal(
    estimator$fitParameters,
    fitParams
  )
  #...
  
  # check the function that results the value from a list
  val <- estimator$itemOrDefaults(list(param=1, test=3), 'param', default = NULL) 
  testthat::expect_equal(val, 1)
  val <- estimator$itemOrDefaults(list(param=1, test=3), 'paramater', default = NULL) 
  testthat::expect_true(is.null(val))
  
}
)

if(F){ # curently this does not work
test_that("estimator fitting works", {
  
  fitParams <- list()
  baseModel <- MRCovNN_submodel1 #(kernel_size = 4)
  
  estimator <- Estimator$new(
    baseModel = baseModel,
    modelParameters = list(kernel_size = 4),
    fitParameters = fitParams, 
    device = 'cpu'
  )
  
  # check the fitting
  estimator$fit(dataset, dataset)
  # estimator$fitEpoch(dataset, batchIndex)
  # estimator$finishFit(valAUCs, modelStateDict, valLosses, epoch)
  # estimator$score(dataset, batchIndex)
  testthat::expect_true()
  
  estimator$fitWholeTrainingSet(dataset)
  
  estimator$save(path, name)
  
  estimator$predictProba(dataset) 
  
  estimator$predict(dataset)

  
  # not sure how to test: batchToDevice(batch)
})
}

test_that("early stopping works", {
  
  earlyStop <- EarlyStopping$new(patience=3, delta=0)
  testthat::expect_true(is.null(earlyStop$bestScore))
  testthat::expect_false(earlyStop$earlyStop)
  earlyStop$call(0.5)
  testthat::expect_equal(earlyStop$bestScore,0.5)
  earlyStop$call(0.4)
  earlyStop$call(0.4)
  earlyStop$call(0.4)
  testthat::expect_true(earlyStop$earlyStop)
  
})
