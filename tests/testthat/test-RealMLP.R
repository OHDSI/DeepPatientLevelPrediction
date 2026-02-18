test_that("setRealMLP settings are created correctly", {
  settings <- setRealMLP(
    numLayers = 1L,
    sizeHidden = 32L,
    dropout = 0.2,
    sizeEmbedding = 16L,
    labelSmoothing = 0.05
  )

  expect_s3_class(settings, "modelSettings")
  expect_equal(settings$modelType, "RealMLP")
  expect_equal(settings$fitFunction, "DeepPatientLevelPrediction::fitEstimator")
  expect_equal(settings$estimatorSettings$beta2, 0.95)
  expect_equal(settings$estimatorSettings$eps, 1e-8)
  expect_equal(settings$estimatorSettings$lrSchedule, "coslog4")
  expect_equal(settings$estimatorSettings$dropoutSchedule, "flat_cos")
  expect_equal(settings$estimatorSettings$weightDecaySchedule, "flat_cos")
  expect_equal(settings$estimatorSettings$scalingLrMult, 6.0)
  expect_equal(settings$estimatorSettings$biasLrMult, 0.1)
  expect_equal(settings$estimatorSettings$actLrMult, 0.1)
  expect_equal(settings$estimatorSettings$embeddingLrMult, 0.1)
  expect_equal(settings$estimatorSettings$biasWdFactor, 0.0)
  expect_true(settings$estimatorSettings$dataDependentInit)
  expect_equal(settings$estimatorSettings$dataDependentInitBatches, 8L)
  expect_equal(settings$estimatorSettings$dataDependentInitBiasMode, "he5")
  expect_equal(settings$estimatorSettings$dataDependentInitBiasScale, 1.0)
  expect_true(settings$param[[1]]$paperMode)
  expect_equal(settings$param[[1]]$tokenAggregation, "sum")
  expect_equal(settings$param[[1]]$featureScaleMode, "scalar")
})

test_that("RealMLP module and schedules are wired correctly", {
  schedules <- reticulate::import_from_path("schedules", path = path)
  t <- 0.31
  expected <- 0.5 * (1.0 - cos(2.0 * pi * log2(1.0 + (2^4 - 1.0) * t)))
  expect_equal(schedules$coslog_k(t, 4L), expected, tolerance = 1e-12)
  expect_equal(schedules$flat_cos(0.25), 1.0, tolerance = 1e-12)
  expect_lt(schedules$flat_cos(0.75), 1.0)

  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  model <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1
  )

  linear <- model$blocks$`__getitem__`(0L)
  bn <- model$blocks$`__getitem__`(1L)
  expect_equal(linear$`_ntp_factor`$item(), 1.0 / sqrt(8), tolerance = 1e-6)
  expect_equal(reticulate::py_to_r(bn$`__class__`$`__name__`), "BatchNorm1d")
  expect_equal(
    reticulate::py_to_r(model$feature_scale$`__class__`$`__name__`),
    "TokenFeatureScale"
  )
})

test_that("RealMLP paper mode resolves aggregation and sum_len_norm works", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  model <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    paper_mode = TRUE,
    token_aggregation = "mean",
    feature_scale_mode = "vector"
  )
  expect_equal(reticulate::py_to_r(model$token_aggregation), "sum")

  modelNorm <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    token_aggregation = "sum_len_norm"
  )
  featureIds <- torch$tensor(
    matrix(c(1, 2, 0, 0, 1, 2, 3, 4), nrow = 2, byrow = TRUE),
    dtype = torch$long
  )
  mask <- (featureIds != 0L)$to(dtype = torch$float32)$unsqueeze(-1L)
  scaled <- torch$ones(c(2L, 4L, 8L)) * mask
  aggregated <- modelNorm$`_aggregate_tokens`(scaled, featureIds)
  expect_equal(
    as.array(aggregated$detach()$cpu()$numpy()),
    array(1, dim = c(2, 8))
  )
})

test_that("RealMLP data-dependent initialization rescales hidden layers", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  model <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1
  )

  linear <- model$blocks$`__getitem__`(0L)
  weight_before <- as.array(linear$weight$detach()$cpu()$numpy())
  vocab_size <- as.integer(
    reticulate::py_to_r(dataset$get_feature_info()$get_vocabulary_size())
  )
  batch <- list(
    feature_ids = torch$randint(1L, vocab_size + 1L, c(32L, 16L), dtype = torch$long),
    feature_values = torch$rand(c(32L, 16L))
  )

  model$data_dependent_init(list(batch), bias_mode = "zero")

  weight_after <- as.array(linear$weight$detach()$cpu()$numpy())
  x <- model$embedding(batch)
  x <- x * model$feature_scale(batch$feature_ids)
  x <- torch$mean(x, dim = 1L)
  z <- linear(x)
  layer_mean <- as.numeric(torch$mean(z)$item())
  expect_false(isTRUE(all.equal(weight_before, weight_after)))
  expect_equal(layer_mean, 0.0, tolerance = 1e-3)
})

test_that("RealMLP data-dependent he5 bias fitting moves mean to target", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  model <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1
  )

  linear <- model$blocks$`__getitem__`(0L)
  vocab_size <- as.integer(
    reticulate::py_to_r(dataset$get_feature_info()$get_vocabulary_size())
  )
  batch <- list(
    feature_ids = torch$randint(1L, vocab_size + 1L, c(32L, 16L), dtype = torch$long),
    feature_values = torch$rand(c(32L, 16L))
  )

  model$data_dependent_init(
    list(batch),
    bias_mode = "he5",
    bias_scale = 1.0
  )

  x <- model$embedding(batch)
  x <- x * model$feature_scale(batch$feature_ids)
  x <- torch$mean(x, dim = 1L)
  z <- linear(x)
  layer_mean <- as.numeric(torch$mean(z)$item())
  target <- 5.0 / sqrt(8.0)
  expect_equal(layer_mean, target, tolerance = 0.1)
})

test_that("RealMLP optimizer parameter groups and dynamic schedules work", {
  settings <- setRealMLP(
    numLayers = 1L,
    sizeHidden = 16L,
    dropout = 0.2,
    sizeEmbedding = 16L,
    labelSmoothing = 0.1,
    device = "cpu"
  )
  settings$estimatorSettings$epochs <- 1L
  settings$estimatorSettings$batchSize <- 64L

  parameters <- list(
    modelParameters = c(
      list(feature_info = dataset$get_feature_info(), modelType = "RealMLP"),
      settings$param[[1]]
    ),
    estimatorSettings = settings$estimatorSettings
  )

  estimator <- createEstimator(parameters = parameters)
  groups <- estimator$optimizer$param_groups
  groupNames <- vapply(groups, function(x) reticulate::py_to_r(x$name), character(1))
  wdFactors <- vapply(groups, function(x) reticulate::py_to_r(x$wd_factor), numeric(1))
  lrFactors <- vapply(groups, function(x) reticulate::py_to_r(x$lr_factor), numeric(1))

  expect_true("scale" %in% groupNames)
  expect_true("embed" %in% groupNames)
  expect_true("bias" %in% groupNames)
  expect_true("act" %in% groupNames)
  expect_true(any(wdFactors == 0.0))
  expect_true(any(lrFactors == settings$estimatorSettings$scalingLrMult))
  expect_true(any(lrFactors == settings$estimatorSettings$embeddingLrMult))

  estimator$total_steps <- 10L
  estimator$global_step <- 1L
  estimator$`_apply_realmlp_step_hparams`()
  lr0 <- reticulate::py_to_r(estimator$optimizer$param_groups[[1]]$lr)

  estimator$global_step <- 5L
  estimator$`_apply_realmlp_step_hparams`()
  lr1 <- reticulate::py_to_r(estimator$optimizer$param_groups[[1]]$lr)
  expect_false(isTRUE(all.equal(lr0, lr1)))
})

test_that("Estimator tie-breaking chooses the last best epoch", {
  estimatorModule <- reticulate::import_from_path("Estimator", path = path)
  expect_equal(estimatorModule$select_best_epoch(list(0.1, 0.5, 0.5), "max"), 2L)
  expect_equal(estimatorModule$select_best_epoch(list(0.2, 0.1, 0.1), "min"), 2L)
})
