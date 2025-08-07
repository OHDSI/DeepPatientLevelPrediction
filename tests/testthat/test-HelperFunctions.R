test_that("Helper functions", {
  expect_true(checkInStringVector("test", c("test", "test2")))
  expect_error(checkInStringVector("test", c("test2", "test3")))

  local({
    filePath <<- withr::local_tempfile(lines = "File contents")
    expect_true(checkFileExists(filePath))
  })
  expect_error(checkFileExists(filePath))

  # checkHigherEqual
  expect_true(checkHigherEqual(2, 1))
  expect_true(checkHigherEqual(1, 1))
  expect_error(checkHigherEqual(0, 1))

  # checkHigher
  expect_true(checkHigher(2, 1))
  expect_error(checkHigher(1, 1))
  expect_error(checkHigher(0, 1))

  # checkIsClass)
  expect_true(checkIsClass(2, c("numeric", "integer")))
  expect_error(checkIsClass(2, c("character")))
  expect_error(checkIsClass("2", c("numeric", "integer")))
})

test_that("r_to_py() converts a data.frame to a polars DataFrame", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_module_available("polars"),
    "Python module `polars` not available for testing"
  )

  df <- data.frame(
    dbl = c(1.1, NA, 3.3),
    int = c(1L, 2L, NA_integer_),
    chr = c("a", NA_character_, "c"),
    lgl = c(TRUE, NA, FALSE),
    strAsFactors = FALSE
  )

  pdf <- r_to_py(df, convert = TRUE)

  expect_true(inherits(pdf, "polars.dataframe.frame.DataFrame"))
  shape <- reticulate::py_to_r(pdf$shape) # (rows, cols)
  expect_equal(shape[[1]], nrow(df))
  expect_equal(shape[[2]], ncol(df))
  expect_setequal(reticulate::py_to_r(pdf$columns), names(df))

  dfBack <- reticulate::py_to_r(pdf)
  expect_identical(dfBack, df)
})

test_that("py_to_r() S3 method returns a faithful R data.frame", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(
    reticulate::py_module_available("polars"),
    "Python module `polars` not available for testing"
  )

  pl <- reticulate::import("polars", convert = FALSE)

  df <- data.frame(
    covariateId = c(1002, 2002, 3003),
    analysisId = c(210, 210, 210),
    valueAsConceptId = c(NA_real_, NA_real_, 0),
    covariateName = c("x", "y", "z"),
    stringsAsFactors = FALSE
  )

  pdf <- pl$DataFrame(df)
  res <- reticulate::py_to_r(pdf)
  expect_identical(res, df)
})

test_that("noneToNA() replaces NULLs with correct typed NA", {
  lst <- list(
    dbl = list(1.1, NULL, 3.3),
    int = list(1L, NULL, 3L),
    lgl = list(TRUE, NULL, FALSE),
    chr = list("a", NULL, "c")
  )
  dtypes <- c("Float64", "Int64", "Boolean", "Utf8")

  res <- noneToNA(lst, dtypes)

  expect_identical(res$dbl, c(1.1, NA_real_, 3.3))
  expect_identical(res$int, c(1L, NA_integer_, 3L))
  expect_identical(res$lgl, c(TRUE, NA, FALSE))
  expect_identical(res$chr, c("a", NA_character_, "c"))
})
