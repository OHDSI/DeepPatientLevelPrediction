# Parameter grids built by setTransformer(). These need neither Python nor
# Eunomia, so they run in every check.

test_that("Positional encodings work", {
  # if temporal is FALSE, positional encodings should not be set and accounted
  # for in the hyperparameter search
  results <- setTransformer(
    temporal = FALSE,
    temporalSettings = list(
      positionalEncoding = list(list(name = "A"), list(name = "B"))
    ),
    hyperParamSearch = "grid"
  )

  expect_equal(length(results$param), 1)
  expect_false("positionalEncoding" %in% results$modelParamNames)


  results <- setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      positionalEncoding = NULL,
      maxSequenceLength = 256
    ),
    hyperParamSearch = "grid"
  )
  # if temporal is TRUE and positionalEncoding is NULL, it should not be set
  # and accounted for in the hyperparameter search
  expect_equal(length(results$param), 1)
  expect_false("positionalEncoding" %in% results$modelParamNames)
  expect_false("positionalEncoding" %in% names(results$param[[1]]))



  results <- setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      positionalEncoding = "SinusoidalPE",
      maxSequenceLength = 256
    )
  )

  expect_equal(length(results$param), 1)
  expect_true("positionalEncoding" %in% results$modelParamNames)
  expect_equal(results$param[[1]]$positionalEncoding, list(name = "SinusoidalPE"))

  # Handles a single PE provided as a list
  peConfig <- list(name = "LearnablePE", dropout = 0.25)
  results <- setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      positionalEncoding = peConfig,
      maxSequenceLength = 256
    ),
    hyperParamSearch = "grid"
  )

  expect_equal(length(results$param), 1)
  expect_true("positionalEncoding" %in% results$modelParamNames)
  expect_equal(results$param[[1]]$positionalEncoding, peConfig)

  # Handles multiple PEs provided
  peConfigs <- list(
    list(name = "SinusoidalPE", dropout = 0.1),
    list(name = "LearnablePE", dropout = 0.2)
  )

  results <- setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      positionalEncoding = peConfigs,
      maxSequenceLength = 256
    ),
    hyperParamSearch = "grid"
  )

  expect_equal(length(results$param), 2)
  expect_true("positionalEncoding" %in% results$modelParamNames)
  expect_equal(results$param[[1]]$positionalEncoding, peConfigs[[1]])
  expect_equal(results$param[[2]]$positionalEncoding, peConfigs[[2]])


  # This tests searching over parameters *within* one PE type.
  peConfig <- list(name = "SinusoidalPE", dropout = c(0.1, 0.2))

  results <- setTransformer(
    temporal = TRUE,
    temporalSettings = list(
      positionalEncoding = peConfig,
      maxSequenceLength = 256
    ),
    hyperParamSearch = "grid"
  )

  expect_equal(length(results$param), 2)
  expect_true("positionalEncoding" %in% results$modelParamNames)

  expect_equal(results$param[[1]]$positionalEncoding, list(name = "SinusoidalPE", dropout = 0.1))
  expect_equal(results$param[[2]]$positionalEncoding, list(name = "SinusoidalPE", dropout = 0.2))

  # This tests searching over multiple PEs with parameter values.
  peConfig <- list(
    list(name = "SinusoidalPE", dropout = c(0.0, 0.1)),
    list(name = "LearnablePE", dropout = 0.15)
  )

  results <- setTransformer(
    dimToken = c(64, 128),
    temporal = TRUE,
    temporalSettings = list(
      positionalEncoding = peConfig,
      maxSequenceLength = 256
    ),
    hyperParamSearch = "grid"
  )

  # Expected number of combinations = 2 (for dimToken) * (2 + 1) (for PEs) = 6
  expect_equal(length(results$param), 6)
  expect_true("positionalEncoding" %in% results$modelParamNames)

  # Spot-check some of the combinations to ensure correctness
  paramSummary <- sapply(results$param, function(p) {
    paste0("dimToken=", p$dimToken, ", PE_name=", p$positionalEncoding$name, ", dropout=", p$positionalEncoding$dropout)
  })

  # Expected combinations
  expectedSummary <- c(
    "dimToken=64, PE_name=SinusoidalPE, dropout=0",
    "dimToken=128, PE_name=SinusoidalPE, dropout=0",
    "dimToken=64, PE_name=SinusoidalPE, dropout=0.1",
    "dimToken=128, PE_name=SinusoidalPE, dropout=0.1",
    "dimToken=64, PE_name=LearnablePE, dropout=0.15",
    "dimToken=128, PE_name=LearnablePE, dropout=0.15"
  )

  # Use sort to make the comparison order-independent
  expect_equal(sort(paramSummary), sort(expectedSummary))
})
