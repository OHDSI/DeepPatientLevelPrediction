resNet <- reticulate::import_from_path("ResNet", path)$ResNet

test_that("LR scheduler that changes per batch works", {
  model <- resNet(
    feature_info = dataset$get_feature_info(),
    size_embedding = 32L, size_hidden = 64L,
    num_layers = 1L, hidden_factor = 1L
  )
  optimizer <- torch$optim$AdamW(model$parameters(), lr = 1e-7)


  exponentialSchedulerPerBatch <-
    reticulate::import_from_path("LrFinder",
      path = path
    )$ExponentialSchedulerPerBatch
  scheduler <- exponentialSchedulerPerBatch(optimizer,
    end_lr = 1e-2,
    num_iter = 5
  )
  expect_equal(scheduler$last_epoch, 0)
  expect_equal(scheduler$optimizer$param_groups[[1]]$lr, 1e-7)

  for (i in 1:5) {
    optimizer$step()
    scheduler$step()
  }

  expect_equal(scheduler$last_epoch, 5)
  expect_equal(
    scheduler$optimizer$param_groups[[1]]$lr,
    (1e-7 * (0.01 / 1e-7)^(5 / 4))
  )
})


test_that("LR finder works", {
  estimatorSettings <- setEstimator(
    batchSize = 32L,
    seed = 42
  )
  modelParameters <- list(
    feature_info = dataset$get_feature_info(),
    size_embedding = 32L,
    size_hidden = 64L,
    num_layers = 1L,
    hidden_factor = 1L,
    modelType = "ResNet"
  )
  estimatorSettings <- estimatorSettings
  lrSettings <- list(
    minLr = 1e-8,
    maxLr = 0.01,
    numLr = 20L,
    divergenceThreshold = 1.1
  )
  # initial LR should be the minLR
  parameters <- list(
    modelParameters = modelParameters,
    estimatorSettings = estimatorSettings
  )
  lr <- getLR(
    parameters = parameters,
    lrSettings = lrSettings,
    dataset = dataset
  )
  tol <- 1e-10
  expect_lte(lr, 0.01 + tol)
  expect_gte(lr, 1e-08 - tol)

  # following the LR finder, the actual training should use the LR found
  if (!dir.exists(file.path(tempdir(), "lrFinder"))) {
    dir.create(file.path(tempdir(), "lrFinder"))
  }
  modelSettings <- setResNet(
    numLayers = 1, sizeHidden = 16, hiddenFactor = 1,
    residualDropout = 0, hiddenDropout = 0,
    sizeEmbedding = 16, hyperParamSearch = "random",
    randomSample = 1,
    setEstimator(
      epochs = 1,
      learningRate = "auto"
    )
  )
  results <- fitEstimator(
    trainData$Train,
    modelSettings, 1,
    file.path(tempdir(), "lrFinder")
  )
  expect_false(results$trainDetails$finalModelParameters$learnSchedule$LRs == 3e-4)
})

test_that("LR finder works with device specified by a function", {
  deviceFun <- function() {
    dev <- "cpu"
    dev
  }
  modelParameters <-
    list(
      feature_info = dataset$get_feature_info(),
      size_embedding = 8L,
      size_hidden = 16L,
      num_layers = 1L,
      hidden_factor = 1L,
      modelType = "ResNet"
    )
  estimatorSettings <- setEstimator(
    batchSize = 32L,
    seed = 42,
    device = deviceFun
  )
  lrSettings <- list(
    minLr = 1e-6,
    maxLr = 0.03,
    numLr = 20L,
    divergenceThreshold = 1.1
  )
  parameters <- list(
    modelParameters = modelParameters,
    estimatorSettings = estimatorSettings
  )
  lr <- getLR(
    parameters = parameters,
    lrSettings = lrSettings,
    dataset = dataset
  )

  tol <- 1e-8
  expect_lte(lr, 0.03 + tol)
  expect_gte(lr, 1e-6 - tol)
})


logspace <- function(start, stop, n) {
  10^(seq(log10(start), log10(stop), length.out = n))
}
sel <- reticulate::import_from_path("LrFinder", path)$suggest_lr

test_that("suggest_lr prefers final descent into the minimum", {
  set.seed(0)
  n <- 120L
  lrs <- logspace(1e-7, 1e-1, n)
  x <- log(lrs)

  s1 <- as.integer(0.12 * n) # end of early drop
  s2 <- as.integer(0.60 * n) # start of final descent

  y <- numeric(n)
  # Segment 1: early steep drop
  y[1:s1] <- 5.0 - 2.8 * (x[1:s1] - x[1]) / (x[s1] - x[1] + 1e-12)
  # Segment 2: slight rise / plateau
  y[(s1 + 1):s2] <- y[s1] + 0.2 * (x[(s1 + 1):s2] - x[s1])
  # Segment 3: later descent to global min
  y[(s2 + 1):n] <- y[s2] - 1.6 * (x[(s2 + 1):n] - x[s2]) / (x[n] - x[s2] + 1e-12)

  # small noise
  y <- y + rnorm(n, sd = 0.02)
  y <- torch$as_tensor(y, dtype = torch$float32)
  lrs <- torch$as_tensor(lrs, dtype = torch$float32)

  res <- sel(lrs, y, return_details = TRUE)
  sug <- res[[1]]
  det <- res[[2]]
  pickIdx <- det$pick_idx
  gmin <- det$gmin

  expect_true(pickIdx >= (s2 - 2))
  expect_lt(pickIdx, gmin)
  expect_true(sug >= lrs[s2]$item() && sug <= lrs[gmin]$item())
})

test_that("suggest_lr on monotonic decrease picks near the end", {
  n <- 80L
  lrs <- logspace(1e-6, 1e-1, n)
  x <- log(lrs)
  losses <- 3.0 - 2.0 * (x - x[1]) / (x[n] - x[1]) # smooth decrease

  res <- sel(lrs, losses, return_details = TRUE)
  det <- res[[2]]

  expect_gte(det$pick_idx, as.integer(0.8 * n) - 5L)
  expect_lt(det$pick_idx, det$gmin)
})


test_that("suggest_lr on monotonic increase falls back to small LR", {
  n <- 60L
  lrs <- logspace(1e-7, 1e-3, n)
  losses <- seq(1.0, 4.0, length.out = n)

  res <- sel(torch$as_tensor(lrs), torch$as_tensor(losses), return_details = TRUE)
  sug <- res[[1]]
  det <- res[[2]]

  # since min is at index 1, fallback should be <= that min LR (or very close)
  expect_lte(sug, lrs[1] * 1.01)
  expect_true(det$reason %in% c("ok", "empty_search", "too_close_to_min"))
})

test_that("suggest_lr handles NaN/Inf and noise", {
  set.seed(1)
  n <- 100L
  lrs <- logspace(1e-6, 1e-2, n)
  x <- log(lrs)

  losses <- 2.0 - 0.8 * (x - x[1]) / (x[n] - x[1])
  losses <- losses + rnorm(n, sd = 0.03)
  losses[10] <- NaN
  losses[25] <- Inf

  res <- sel(torch$as_tensor(lrs), torch$as_tensor(losses), return_details = TRUE)
  sug <- res[[1]]
  det <- res[[2]]

  expect_true(is.finite(sug))
  expect_true(det$pick_idx >= 1 && det$pick_idx < length(det$lrs))
})

test_that("suggest_lr is robust to unsorted LR input", {
  set.seed(2)
  n <- 90L
  lrs <- logspace(1e-7, 1e-2, n)
  x <- log(lrs)
  # U-shaped curve with minimum near 70% of range
  losses <- 1.5 + 0.5 * (x - mean(x))^2
  losses <- losses + rnorm(n, sd = 0.01)

  perm <- sample.int(n)
  lrsPerm <- lrs[perm]
  lossesPerm <- losses[perm]

  res1 <- sel(torch$as_tensor(lrs), torch$as_tensor(losses), return_details = TRUE)
  res2 <- sel(torch$as_tensor(lrsPerm), torch$as_tensor(lossesPerm), return_details = TRUE)

  sug1 <- res1[[1]]
  sug2 <- res2[[1]]

  # suggestions should be close (within a small multiplicative factor)
  expect_lt(abs(log(sug1) - log(sug2)), 0.2) # within ~e^0.2 ≈ 1.22x
})

test_that("suggest_lr handles very short inputs", {
  # One point
  lrs <- 1e-4
  losses <- 1.0
  res <- sel(lrs, losses, return_details = TRUE)
  expect_equal(res[[1]], lrs[[1]])

  # Two points
  lrs <- c(-1e-4, 2e-4)
  losses <- c(1.0, 0.8)
  res <- sel(lrs, losses, return_details = TRUE)
  expect_true(res[[1]] %in% lrs)
})

test_that("suggest_lr is (roughly) invariant to affine loss scaling", {
  set.seed(3)
  n <- 100L
  lrs <- logspace(1e-7, 1e-1, n)
  x <- log(lrs)
  losses <- 2.0 - 1.2 * (x - x[1]) / (x[n] - x[1]) + rnorm(n, sd = 0.02)

  res1 <- sel(lrs, losses, return_details = TRUE)
  res2 <- sel(lrs, 3.7 * losses + 0.5, return_details = TRUE)

  sug1 <- res1[[1]]
  sug2 <- res2[[1]]

  # allow some tolerance; compare in log space
  expect_lt(abs(log(sug1) - log(sug2)), 0.3)
})
