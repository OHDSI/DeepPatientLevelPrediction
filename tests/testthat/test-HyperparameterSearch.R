# Hyperparameter search: the tuning metric, PLP's search settings, and
# recovery from a combination that runs out of memory.

test_that("the tuning metric picks the best hyperparameters", {
  results <- list(
    list(gridPerformance = list(cvPerformance = 0.2), prediction = "a"),
    NULL,
    list(gridPerformance = list(cvPerformance = 0.9), prediction = "c")
  )
  expect_equal(getBestGridIndex(results, maximize = TRUE), 3)
  expect_equal(getBestGridIndex(results, maximize = FALSE), 1)

  cache <- trainingCache$new(withr::local_tempdir())
  trimmed <- cache$trimPerformance(results, maximize = FALSE)
  expect_equal(trimmed[[1]]$prediction, "a")
  expect_equal(trimmed[[3]]$prediction, list(NULL))
})

test_that("search settings from PatientLevelPrediction are reported as ignored", {
  expect_warning(
    ignored <- warnIgnoredSearchSettings(
      PatientLevelPrediction::createHyperparameterSettings(
        search = "random", sampleSize = 2, randomSeed = 1
      )
    ),
    "ignores `search`, `sampleSize`, `randomSeed`"
  )
  expect_equal(ignored, c("search", "sampleSize", "randomSeed"))
  metric <- PatientLevelPrediction::createTuningMetric(
    fun = function(prediction) 1, maximize = FALSE, name = "constant"
  )
  expect_no_warning(warnIgnoredSearchSettings(
    PatientLevelPrediction::createHyperparameterSettings(tuningMetric = metric)
  ))
  expect_no_warning(warnIgnoredSearchSettings(NULL))
})

test_that("computeTuningPerformance uses the requested metric", {
  metric <- PatientLevelPrediction::createTuningMetric(
    fun = function(prediction) mean(prediction$value),
    maximize = FALSE,
    name = "meanRisk"
  )
  prediction <- data.frame(
    rowId = 1:4, outcomeCount = c(0, 1, 0, 1),
    value = c(0.1, 0.2, 0.3, 0.4), index = c(1, 1, 2, 2)
  )
  performance <- computeTuningPerformance(prediction, list(sizeHidden = 4), metric)
  expect_equal(performance$cvPerformance, 0.25)
  expect_equal(performance$metric, "meanRisk")
  expect_true(all(performance$hyperSummary$metric == "meanRisk"))
  expect_equal(
    computeTuningPerformance(prediction, list(sizeHidden = 4))$metric,
    "computeAuc"
  )
})

test_that("gridCvDeep selects the hyperparameters with the requested metric", {
  skip_if_no_python()
  data <- createMockData()
  withr::defer(Andromeda::close(data))
  labels <- createMockLabels()
  settings <- setMultiLayerPerceptron(
    numLayers = 1, sizeHidden = c(4, 8), dropout = 0, sizeEmbedding = 4,
    estimatorSettings = setEstimator(
      learningRate = 0.01, batchSize = 4, epochs = 1, device = "cpu", seed = 42
    ),
    hyperParamSearch = "grid"
  )
  metric <- PatientLevelPrediction::createTuningMetric(
    fun = function(prediction) stop("not used with the mocked cross validation"),
    maximize = FALSE,
    name = "lowerIsBetter"
  )
  performance <- c(0.9, 0.1)
  fit <- function(tuningMetric) {
    testthat::with_mocked_bindings(
      gridCvDeep(
        mappedData = data, labels = labels, modelSettings = settings,
        modelLocation = withr::local_tempdir(.local_envir = parent.frame(3)),
        analysisPath = withr::local_tempdir(.local_envir = parent.frame(3)),
        tuningMetric = tuningMetric
      ),
      doCrossValidation = function(dataset, labels, parameters, modelSettings, tuningMetric) {
        prediction <- labels
        prediction$value <- 0.5
        index <- match(parameters$sizeHidden, c(4, 8))
        list(
          prediction = prediction,
          param = c(parameters, list(learnSchedule = list(LRs = list(0.01)))),
          gridPerformance = list(
            cvPerformance = performance[index],
            hyperSummary = data.frame(metric = "x")
          )
        )
      }
    )
  }
  expect_equal(fit(metric)$finalParam$sizeHidden, 8)
  expect_equal(fit(NULL)$finalParam$sizeHidden, 4)
})

test_that("memory cleanup leaves tensors that are still in use intact", {
  skip_if_no_python()
  importModule("gpu_memory_cleanup")
  # Pretend that GPU memory use stays high after garbage collection and that a
  # tensor still in use is on the GPU, which is when cleanup used to free the
  # storage of every tensor it found.
  result <- reticulate::py_run_string("
def cleanup_with_high_memory_use():
    import warnings
    import torch
    import gpu_memory_cleanup as cleanup

    live = torch.arange(4.0)
    saved = {name: getattr(torch.cuda, name) for name in ('memory_allocated', 'empty_cache')}
    saved_get_tensors = getattr(cleanup, 'get_tensors', None)
    torch.cuda.memory_allocated = lambda *args, **kwargs: 4 * 1024 ** 3
    torch.cuda.empty_cache = lambda: None
    cleanup.get_tensors = lambda gpu_only=True: iter([live])
    try:
        # warnings from the cleanup code are beside the point here
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cleanup.memory_cleanup()
    finally:
        for name, value in saved.items():
            setattr(torch.cuda, name, value)
        if saved_get_tensors is None:
            del cleanup.get_tensors
        else:
            cleanup.get_tensors = saved_get_tensors
    return live.untyped_storage().nbytes() // live.element_size()

remaining = cleanup_with_high_memory_use()
", local = TRUE)
  expect_equal(result$remaining, 4)
})

test_that("the search frees a combination that runs out of memory and continues", {
  skip_if_no_python()
  reticulate::py_run_string("
import gc
import weakref
import torch

failed_models = []

def run_out_of_memory():
    model = torch.nn.Linear(2, 1)  # stands in for a model on the GPU
    failed_models.append(weakref.ref(model))
    raise torch.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB')

def failed_model_freed():
    gc.collect()
    return failed_models[-1]() is None
")
  main <- reticulate::import_main()
  labels <- createMockLabels()
  result <- testthat::with_mocked_bindings(
    doCrossValidation(
      dataset = NULL, labels = labels, parameters = list(numLayers = 1),
      modelSettings = list()
    ),
    doCrossValidationImpl = function(...) main$run_out_of_memory()
  )
  expect_true(all(is.na(result$prediction$value)))
  expect_equal(nrow(result$prediction), nrow(labels))
  expect_true(main$failed_model_freed())
})
