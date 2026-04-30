test_that("setRealMLP settings are created correctly", {
  settings <- setRealMLP(
    numLayers = 1L,
    sizeHidden = 32L,
    dropout = 0.2,
    sizeEmbedding = 16L,
    labelSmoothing = 0.05,
    numericEmbeddingMode = "pbld",
    numericNumFrequencies = 6L,
    numericPeriodicInitStd = 0.2,
    numericPbldHiddenDim = 12L,
    numericPbldEmbeddingDim = 5L
  )

  expect_s3_class(settings, "modelSettings")
  expect_equal(settings$modelType, "RealMLP")
  expect_equal(settings$fitFunction, "DeepPatientLevelPrediction::fitEstimator")
  expect_equal(settings$estimatorSettings$beta2, 0.95)
  expect_equal(settings$estimatorSettings$eps, 1e-8)
  expect_equal(settings$estimatorSettings$metric, "loss")
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
  expect_equal(settings$estimatorSettings$dataDependentInitMode, "paper_lsuv")
  expect_equal(settings$estimatorSettings$dataDependentInitTargetVar, 1.0)
  expect_equal(settings$estimatorSettings$dataDependentInitMaxRows, 65536L)
  expect_equal(settings$estimatorSettings$dataDependentInitGainClip, 10.0)
  expect_equal(settings$estimatorSettings$dataDependentInitBiasMode, "he5")
  expect_equal(settings$estimatorSettings$dataDependentInitBiasScale, 1.0)
  expect_equal(settings$estimatorSettings$dataDependentInitBiasRefitSteps, 2L)
  expect_true(settings$param[[1]]$paperMode)
  expect_equal(settings$param[[1]]$numericEmbeddingMode, "pbld")
  expect_equal(settings$param[[1]]$numericNumFrequencies, 6L)
  expect_equal(settings$param[[1]]$numericPeriodicInitStd, 0.2)
  expect_equal(settings$param[[1]]$numericPbldHiddenDim, 12L)
  expect_equal(settings$param[[1]]$numericPbldEmbeddingDim, 5L)
  expect_equal(settings$param[[1]]$tokenAggregation, "sum")
  expect_equal(settings$param[[1]]$featureScaleMode, "scalar")
})

test_that("setRealMLP expands vector parameters into grid combinations", {
  settings <- setRealMLP(
    numLayers = c(1L, 2L),
    sizeHidden = c(16L, 32L),
    dropout = c(0.1, 0.2),
    sizeEmbedding = 8L,
    labelSmoothing = c(0, 0.1),
    numericEmbeddingMode = c("scale", "pbld"),
    paperMode = c(TRUE, FALSE),
    tokenAggregation = "auto",
    featureScaleMode = "auto"
  )

  expect_equal(length(settings$param), 64L)
  expect_true(all(vapply(settings$param, function(x) length(x$numLayers), integer(1)) == 1L))
  expect_true(all(vapply(settings$param, function(x) length(x$sizeHidden), integer(1)) == 1L))
  expect_true(all(vapply(settings$param, function(x) length(x$dropout), integer(1)) == 1L))
  expect_true(all(vapply(settings$param, function(x) length(x$numericEmbeddingMode), integer(1)) == 1L))
  expect_true(all(vapply(settings$param, function(x) length(x$estimator.labelSmoothing), integer(1)) == 1L))
  expect_setequal(
    vapply(settings$param, function(x) x$estimator.labelSmoothing, numeric(1)),
    c(0, 0.1)
  )
  expect_false(any(vapply(settings$param, function(x) x$tokenAggregation == "auto", logical(1))))
  expect_false(any(vapply(settings$param, function(x) x$featureScaleMode == "auto", logical(1))))
  expect_true(all(
    vapply(settings$param, function(x) {
      if (isTRUE(x$paperMode)) {
        x$tokenAggregation == "sum"
      } else {
        x$tokenAggregation == "mean"
      }
    }, logical(1))
  ))
  expect_true(all(vapply(settings$param, function(x) x$featureScaleMode == "scalar", logical(1))))
})

test_that("setRealMLP random search samples expanded grid combinations", {
  settings <- setRealMLP(
    numLayers = c(1L, 2L),
    sizeHidden = c(16L, 32L),
    dropout = c(0.1, 0.2),
    sizeEmbedding = 8L,
    hyperParamSearch = "random",
    randomSample = 3L,
    randomSampleSeed = 42L
  )

  expect_equal(length(settings$param), 3L)
  expect_error(
    setRealMLP(
      numLayers = 1L,
      sizeHidden = 16L,
      dropout = 0.1,
      sizeEmbedding = 8L,
      hyperParamSearch = "random",
      randomSample = 2L
    )
  )
})

test_that("setRealMLP validates vector-aware RealMLP parameters", {
  expect_error(
    setRealMLP(labelSmoothing = 1.1),
    "labelSmoothing needs to be <= 1"
  )
  expect_error(
    setRealMLP(numericEmbeddingMode = c("scale", "invalid")),
    "numericEmbeddingMode has incorrect value"
  )
  expect_error(
    setRealMLP(dataDependentInitMode = c("paper_lsuv", "invalid")),
    "dataDependentInitMode has incorrect value"
  )
  expect_error(
    setRealMLP(tokenAggregation = c("mean", "invalid")),
    "tokenAggregation has incorrect value"
  )
  expect_error(
    setRealMLP(featureScaleMode = c("scalar", "invalid")),
    "featureScaleMode has incorrect value"
  )
})

test_that("RealMLP supports PL and PBLD numerical embedding modes", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  featureInfo <- dataset$get_feature_info()
  vocab_size <- as.integer(reticulate::py_to_r(featureInfo$get_vocabulary_size()))
  batch <- list(
    feature_ids = torch$randint(1L, vocab_size + 1L, c(8L, 10L), dtype = torch$long),
    feature_values = torch$randn(c(8L, 10L), dtype = torch$float32)
  )

  modelPl <- realMlp(
    feature_info = featureInfo,
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    numeric_embedding_mode = "pl"
  )
  outPl <- modelPl(batch)
  expect_equal(as.integer(reticulate::py_to_r(outPl$size(0L))), 8L)
  expect_false(any(is.na(as.array(outPl$detach()$cpu()$numpy()))))

  modelPbld <- realMlp(
    feature_info = featureInfo,
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    numeric_embedding_mode = "pbld"
  )
  outPbld <- modelPbld(batch)
  expect_equal(as.integer(reticulate::py_to_r(outPbld$size(0L))), 8L)
  expect_false(any(is.na(as.array(outPbld$detach()$cpu()$numpy()))))
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
  expect_true(reticulate::py_to_r(model$use_two_logit_ce))
  expect_equal(reticulate::py_to_r(model$output_dim), 2L)

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
  expected <- rbind(
    rep(sqrt(2), 8),
    rep(2, 8)
  )
  expect_equal(
    as.array(aggregated$detach()$cpu()$numpy()),
    expected,
    tolerance = 1e-6
  )
})

test_that("RealMLP paper mode uses CE-compatible outputs and probabilities", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  model <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    paper_mode = TRUE
  )

  vocab_size <- as.integer(
    reticulate::py_to_r(dataset$get_feature_info()$get_vocabulary_size())
  )
  batch <- list(
    feature_ids = torch$randint(1L, vocab_size + 1L, c(8L, 12L), dtype = torch$long),
    feature_values = torch$rand(c(8L, 12L))
  )
  logits <- model(batch)
  expect_equal(as.integer(reticulate::py_to_r(logits$size(1L))), 2L)

  targets <- torch$tensor(c(0, 1, 1, 0, 1, 0, 0, 1), dtype = torch$float32)
  loss <- model$compute_loss(logits, targets, criterion = NULL, label_smoothing = 0.1)
  expect_gt(as.numeric(loss$item()), 0.0)

  probs <- model$predict_proba_from_output(logits)
  expect_equal(as.integer(reticulate::py_to_r(probs$size(0L))), 8L)
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

test_that("RealMLP paper_lsuv data-dependent init mode runs with row caps", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  model <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1
  )
  vocab_size <- as.integer(
    reticulate::py_to_r(dataset$get_feature_info()$get_vocabulary_size())
  )
  batch <- list(
    feature_ids = torch$randint(1L, vocab_size + 1L, c(64L, 12L), dtype = torch$long),
    feature_values = torch$rand(c(64L, 12L))
  )

  expect_no_error(
    model$data_dependent_init(
      list(batch),
      init_mode = "paper_lsuv",
      target_var = 1.0,
      max_rows = 16L,
      gain_clip = 5.0,
      bias_mode = "zero",
      bias_refit_steps = 1L
    )
  )
})

test_that("RealMLP optimizer parameter groups and dynamic schedules work", {
  settings <- setRealMLP(
    numLayers = 1L,
    sizeHidden = 16L,
    dropout = 0.2,
    sizeEmbedding = 16L,
    labelSmoothing = 0.1,
    numericEmbeddingMode = "pbld",
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
  expect_equal(
    reticulate::py_to_r(estimator$model$numeric_embedding_mode),
    "pbld"
  )

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

test_that("RealMLP wide path composes logits as wide + alpha * deep", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  featureInfo <- dataset$get_feature_info()

  modelWide <- realMlp(
    feature_info = featureInfo,
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    paper_mode = FALSE,
    wide_enabled = TRUE,
    wide_alpha_init = 1.0
  )
  modelDeep <- realMlp(
    feature_info = featureInfo,
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    paper_mode = FALSE,
    wide_enabled = FALSE
  )

  # Copy the deep branch state so wide and deep-only outputs are comparable.
  modelDeep$load_state_dict(modelWide$state_dict(), strict = FALSE)
  with_no_grad <- torch$no_grad()
  with_no_grad$`__enter__`()
  modelWide$wide_embedding$weight$zero_()
  modelWide$wide_bias$zero_()
  with_no_grad$`__exit__`(NULL, NULL, NULL)
  modelWide$eval()
  modelDeep$eval()

  vocab_size <- as.integer(reticulate::py_to_r(featureInfo$get_vocabulary_size()))
  batch <- list(
    feature_ids = torch$randint(1L, vocab_size + 1L, c(6L, 10L), dtype = torch$long),
    feature_values = torch$rand(c(6L, 10L))
  )

  outWideAlpha1 <- modelWide(batch)$detach()$cpu()$numpy()
  outDeep <- modelDeep(batch)$detach()$cpu()$numpy()
  expect_equal(as.array(outWideAlpha1), as.array(outDeep), tolerance = 1e-6)

  with_no_grad <- torch$no_grad()
  with_no_grad$`__enter__`()
  modelWide$wide_alpha$fill_(0.0)
  with_no_grad$`__exit__`(NULL, NULL, NULL)
  outWideAlpha0 <- modelWide(batch)$detach()$cpu()$numpy()
  expect_equal(as.array(outWideAlpha0), array(0, dim = dim(outWideAlpha0)), tolerance = 1e-6)
})

test_that("RealMLP wide L1 regularization excludes padding row", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  model <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    paper_mode = FALSE,
    wide_enabled = TRUE,
    l1_wide_lambda = 0.5
  )

  with_no_grad <- torch$no_grad()
  with_no_grad$`__enter__`()
  weight_rows <- as.integer(reticulate::py_to_r(model$wide_embedding$weight$size(0L)))
  weight_matrix <- matrix(0, nrow = weight_rows, ncol = 1)
  # Python index 1 -> R row 2, Python index 2 -> R row 3.
  weight_matrix[2, 1] <- 2.0
  weight_matrix[3, 1] <- -3.0
  model$wide_embedding$weight$copy_(torch$tensor(weight_matrix, dtype = torch$float32))
  with_no_grad$`__exit__`(NULL, NULL, NULL)

  reg <- model$regularization_loss()
  expect_equal(as.numeric(reg$item()), 2.5, tolerance = 1e-8)
})

test_that("RealMLP wide initialization loads coefficients from csv", {
  realMlp <- reticulate::import_from_path("RealMLP", path = path)$RealMLP
  initCsv <- tempfile(fileext = ".csv")
  writeLines(
    c(
      "columnId,weight",
      "1,0.25",
      "2,-0.75",
      "(Intercept),-1.5"
    ),
    con = initCsv
  )

  model <- realMlp(
    feature_info = dataset$get_feature_info(),
    size_embedding = 8L,
    size_hidden = 16L,
    num_layers = 1L,
    dropout = 0.1,
    paper_mode = FALSE,
    wide_enabled = TRUE,
    wide_init_path = initCsv
  )

  weights <- as.array(model$wide_embedding$weight$detach()$cpu()$numpy())
  w1 <- as.numeric(weights[2, 1])
  w2 <- as.numeric(weights[3, 1])
  wb <- as.numeric(model$wide_bias$item())
  expect_equal(w1, 0.25, tolerance = 1e-8)
  expect_equal(w2, -0.75, tolerance = 1e-8)
  expect_equal(wb, -1.5, tolerance = 1e-8)
})
