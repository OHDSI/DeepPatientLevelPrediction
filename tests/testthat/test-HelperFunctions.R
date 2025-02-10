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
