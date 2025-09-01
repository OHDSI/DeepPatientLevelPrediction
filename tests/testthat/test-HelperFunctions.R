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

test_that("expandComponentGrid expands user settings into a flat grid", {
  res <- expandComponentGrid("AdamW")
  expect_type(res, "list")
  expect_length(res, 1)
  expect_equal(as.character(res[[1]]$name), "AdamW")

  res <- expandComponentGrid(list(name = "AdamW", lr = 0.001))
  expect_length(res, 1)
  expect_identical(as.character(res[[1]]$name), "AdamW")
  expect_identical(res[[1]]$lr, 0.001)

  res <- expandComponentGrid(list(
    name = "AdamW",
    lr = c(0.001, 0.0001),
    weight_decay = c(1e-3, 1e-6)
  ))
  expect_length(res, 4)
  combos <- expand.grid(
    lr = c(0.001, 0.0001),
    weight_decay = c(1e-3, 1e-6),
    stringsAsFactors = FALSE
  )
  for (i in seq_len(nrow(combos))) {
    expect_true(
      any(vapply(
        res,
        function(x) {
          identical(as.character(x$name), "AdamW") &&
            identical(x$lr, combos$lr[[i]]) &&
            identical(x$weight_decay, combos$weight_decay[[i]])
        },
        logical(1)
      )),
      info = paste("Missing combo:", combos$lr[[i]], combos$weight_decay[[i]])
    )
  }
  res <- expandComponentGrid(list(
    list(name = "AdamW", lr = c(0.001, 0.0001)),
    list(name = "SGD", lr = c(0.1, 0.01), momentum = c(0.9, 0.95))
  ))
  expect_length(res, 6)
  expect_true(all(vapply(res, function(x) "name" %in% names(x), logical(1))))
  expect_true(any(vapply(res, function(x) identical(as.character(x$name), "AdamW") && x$lr %in% c(0.001, 0.0001), logical(1))))
  expect_true(any(vapply(res, function(x) identical(as.character(x$name), "SGD") && x$lr == 0.1 && x$momentum == 0.9, logical(1))))
  expect_true(any(vapply(res, function(x) identical(as.character(x$name), "SGD") && x$lr == 0.01 && x$momentum == 0.95, logical(1))))
})

test_that("handles a single component name as a string", {
  componentSetting <- "SinusoidalPE"
  result <- expandComponentGrid(componentSetting)

  expect_type(result, "list")
  expect_length(result, 1)
  expect_equal(result[[1]], list(name = "SinusoidalPE"))
})

test_that("handles a single, fully-specified configuration", {
  componentSetting <- list(name = "LearnablePE", dropout = 0.1)
  result <- expandComponentGrid(componentSetting)

  expect_type(result, "list")
  expect_length(result, 1)
  expect_equal(result[[1]], list(name = "LearnablePE", dropout = 0.1))
})

test_that("expands a single vector hyperparameter", {
  componentSetting <- list(name = "SinusoidalPE", dropout = c(0.1, 0.2))
  result <- expandComponentGrid(componentSetting)

  expect_length(result, 2)
  expect_equal(result[[1]], list(name = "SinusoidalPE", dropout = 0.1))
  expect_equal(result[[2]], list(name = "SinusoidalPE", dropout = 0.2))
})

test_that("expands multiple vector hyperparameters (Cartesian product)", {
  componentSetting <- list(name = "AdamW", lr = c(1e-3, 1e-4), weight_decay = c(0.1, 0.01))
  result <- expandComponentGrid(componentSetting)

  expect_length(result, 4)
  expect_equal(result[[1]], list(name = "AdamW", lr = 1e-3, weight_decay = 0.1))
  expect_equal(result[[2]], list(name = "AdamW", lr = 1e-4, weight_decay = 0.1))
  expect_equal(result[[3]], list(name = "AdamW", lr = 1e-3, weight_decay = 0.01))
  expect_equal(result[[4]], list(name = "AdamW", lr = 1e-4, weight_decay = 0.01))
})

test_that("preserves a single nested list as a static config", {
  componentSetting <- list(
    name = "TUPE",
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.1)
  )
  result <- expandComponentGrid(componentSetting)

  expect_length(result, 1)
  expect_equal(result[[1]], componentSetting)
})

test_that("preserves nested list while expanding a hyperparameter", {
  componentSetting <- list(
    name = "TUPE",
    dropout = c(0.1, 0.2), # Hyperparameter to expand
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.0) # Static config
  )
  result <- expandComponentGrid(componentSetting)

  expect_length(result, 2)

  expected1 <- list(
    name = "TUPE",
    dropout = 0.1,
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.0)
  )
  expect_equal(result[[1]], expected1)

  expected2 <- list(
    name = "TUPE",
    dropout = 0.2,
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.0)
  )
  expect_equal(result[[2]], expected2)
})

test_that("preserves multiple nested lists while expanding hyperparameters", {
  componentSetting <- list(
    name = "TemporalPE",
    dropout = c(0.1, 0.2),
    abs_config = list(name = "SinusoidalPE"),
    rel_config = list(name = "RelativePE")
  )
  result <- expandComponentGrid(componentSetting)

  expect_length(result, 2)

  expect_equal(result[[1]]$dropout, 0.1)
  expect_equal(result[[1]]$abs_config, list(name = "SinusoidalPE"))
  expect_equal(result[[1]]$rel_config, list(name = "RelativePE"))
})


test_that("expands a list of multiple component templates", {
  componentSetting <- list(
    list(name = "SinusoidalPE", dropout = 0.1),
    list(name = "LearnablePE", dropout = c(0.2, 0.3))
  )
  result <- expandComponentGrid(componentSetting)

  expect_length(result, 3)

  expect_equal(result[[1]], list(name = "SinusoidalPE", dropout = 0.1))
  expect_equal(result[[2]], list(name = "LearnablePE", dropout = 0.2))
  expect_equal(result[[3]], list(name = "LearnablePE", dropout = 0.3))
})

test_that("expands a list of templates containing nested static configs", {
  componentSetting <- list(
    list(
      name = "TUPE",
      dropout = c(0.1, 0.2),
      base_pe_config = list(name = "SinusoidalPE")
    ),
    list(name = "LearnablePE", dropout = 0.15)
  )
  result <- expandComponentGrid(componentSetting)

  expect_length(result, 3)

  expect_equal(
    result[[1]],
    list(
      name = "TUPE",
      dropout = 0.1,
      base_pe_config = list(name = "SinusoidalPE")
    )
  )
  expect_equal(
    result[[2]], 
    list(
      name = "TUPE", 
      dropout = 0.2, 
      base_pe_config = list(name = "SinusoidalPE")))
  expect_equal(result[[3]], list(name = "LearnablePE", dropout = 0.15))
})

test_that("recursively expands hyperparameters within nested lists", {
  componentSetting <- list(
    name = "TUPE",
    top_level_dropout = c(0.1, 0.2),
    base_pe_config = list(
      name = "SinusoidalPE",
      dropout = c(0.01, 0.02, 0.03)
    )
  )

  result <- expandComponentGrid(componentSetting)

  expect_length(result, 6)

  expect_equal(result[[1]], list(
    name = "TUPE",
    top_level_dropout = 0.1,
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.01)
  ))
  expect_equal(result[[2]], list(
    name = "TUPE",
    top_level_dropout = 0.2,
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.01)
  ))
  expect_equal(result[[3]], list(
    name = "TUPE",
    top_level_dropout = 0.1,
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.02)
  ))
  expect_equal(result[[4]], list(
    name = "TUPE",
    top_level_dropout = 0.2,
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.02)
  ))
  expect_equal(result[[5]], list(
    name = "TUPE",
    top_level_dropout = 0.1,
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.03)
  ))
  expect_equal(result[[6]], list( 
    name = "TUPE",
    top_level_dropout = 0.2,
    base_pe_config = list(name = "SinusoidalPE", dropout = 0.03)
  ))
})

test_that("handles multiple nested hyperparameter lists", {
  componentSetting <- list(
    name = "TemporalPE",
    abs_config = list(name = "LearnablePE", dropout = c(0.1, 0.2)), # 2 values
    rel_config = list(name = "RelativePE", max_relative_position = c(16, 32)) # 2 values
  )
  result <- expandComponentGrid(componentSetting)

  # Expected number of combinations = 2 * 2 = 4
  expect_length(result, 4)

  expect_true(all(sapply(result, function(x) x$name == "TemporalPE")))
  expect_equal(result[[1]], list(
    name = "TemporalPE",
    abs_config = list(name = "LearnablePE", dropout = 0.1),
    rel_config = list(name = "RelativePE", max_relative_position = 16)
  ))
  expect_equal(result[[2]], list(
    name = "TemporalPE",
    abs_config = list(name = "LearnablePE", dropout = 0.2),
    rel_config = list(name = "RelativePE", max_relative_position = 16)
  ))
  expect_equal(result[[3]], list(
    name = "TemporalPE",
    abs_config = list(name = "LearnablePE", dropout = 0.1),
    rel_config = list(name = "RelativePE", max_relative_position = 32)
  ))
  expect_equal(result[[4]], list(
    name = "TemporalPE",
    abs_config = list(name = "LearnablePE", dropout = 0.2),
    rel_config = list(name = "RelativePE", max_relative_position = 32)
  ))

})
test_that("keepDefaults overrides defaults and adds new keys", {
  defaults <- list(lr = 0.01, weight_decay = 0)
  users <- list(lr = 0.001, beta1 = 0.9)

  res <- keepDefaults(defaults, users)

  expect_equal(res$lr, 0.001)
  expect_equal(res$weight_decay, 0)
  expect_true("beta1" %in% names(res))
  expect_equal(res$beta1, 0.9)
})

test_that("keepDefaults returns defaults when userSettings is NULL", {
  defaults <- list(lr = 0.01, weight_decay = 0)
  res <- keepDefaults(defaults, NULL)
  expect_identical(res, defaults)
})

test_that("keepDefaults errors when userSettings is not a list", {
  defaults <- list(lr = 0.01)
  expect_error(keepDefaults(defaults, "not-a-list"), "Settings argument must be a list.")
})
