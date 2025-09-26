test_that("flashEnvCheck aggregates multiple environment failures", {
  fake <- makeFakeTorch(
    cudaAvailable = FALSE,
    cudaVersion = NULL,
    capMajor = 7L,
    capMinor = 5L,
    deviceName = "NVIDIA T4",
    bf16Supported = FALSE
  )
  localMockFlashBindings(fakeTorch = fake, flashModuleAvailable = FALSE)

  env <- flashEnvCheck(
    estimatorSettings = list(device = "cpu", precision = "float32")
  )

  expect_false(env$ok)
  expect_true(any(grepl("device must be 'cuda'", env$reasons)))
  expect_true(any(grepl("torch.cuda.is_available\\(\\) is FALSE", env$reasons)))
  expect_true(any(grepl("torch.version.cuda is NULL", env$reasons)))
  expect_true(any(grepl("flash-attn \\(v2\\) is not importable", env$reasons)))
  expect_true(any(grepl("compute capability >= 8\\.0", env$reasons)))
  expect_true(any(grepl("BF16 is not supported", env$reasons)))
  expect_true(any(grepl("precision must be 'bfloat16'", env$reasons)))

  msg <- formatFlashErrors(
    "Header",
    env$reasons
  )
  expect_true(grepl("^Header\\n  - ", msg))
})

test_that("flashEnvCheck passes when environment is compatible", {
  fake <- makeFakeTorch(
    cudaAvailable = TRUE,
    cudaVersion = "12.1",
    capMajor = 8L,
    capMinor = 0L,
    deviceName = "NVIDIA A100-SXM4-40GB",
    bf16Supported = TRUE
  )
  localMockFlashBindings(fakeTorch = fake, flashModuleAvailable = TRUE)

  env <- flashEnvCheck(
    estimatorSettings = list(device = "cuda", precision = "bfloat16")
  )

  expect_true(env$ok)
  expect_length(env$reasons, 0L)
  expect_equal(env$info$computeCapability, "8.0")
  expect_true(
    is.character(env$info$torchCudaVersion) || is.na(env$info$torchCudaVersion)
  )
})

test_that("flashParamCheck flags headDim divisibility/multiple-of-8/upper-bounds", {
  envInfo <- list(info = list(deviceName = "NVIDIA GeForce RTX 3090"))

  out1 <- flashParamCheck(
    paramCombo = list(dimToken = 190, numHeads = 8),
    temporal = FALSE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_false(out1$ok)
  expect_true(any(grepl("must be divisible by numHeads", out1$reasons)))

  out2 <- flashParamCheck(
    paramCombo = list(dimToken = 180, numHeads = 5),
    temporal = FALSE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_false(out2$ok)
  expect_true(any(grepl("multiple of 8", out2$reasons)))

  out3 <- flashParamCheck(
    paramCombo = list(dimToken = 2112, numHeads = 8),
    temporal = FALSE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_false(out3$ok)
  expect_true(any(grepl("must be <= 256", out3$reasons)))

  out4 <- flashParamCheck(
    paramCombo = list(dimToken = 1632, numHeads = 8),
    temporal = FALSE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_false(out4$ok)
})

test_that("flashParamCheck rejects disallowed PEs and accepts allowed PEs", {
  envInfo <- list(info = list(deviceName = "NVIDIA A100-SXM4-40GB"))

  outRel <- DeepPatientLevelPrediction:::flashParamCheck(
    paramCombo = list(
      dimToken = 192,
      numHeads = 8,
      positionalEncoding = list(name = "RelativePE")
    ),
    temporal = TRUE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_false(outRel$ok)
  expect_true(any(grepl("not supported by flash-attn v2", outRel$reasons)))

  outRoPE <- flashParamCheck(
    paramCombo = list(
      dimToken = 192,
      numHeads = 8,
      positionalEncoding = list(name = "RotaryPE")
    ),
    temporal = TRUE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_true(outRoPE$ok)

  outLearnable <- flashParamCheck(
    paramCombo = list(
      dimToken = 192,
      numHeads = 8,
      positionalEncoding = list(name = "LearnablePE")
    ),
    temporal = TRUE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_true(outLearnable$ok)

  outStoch <- flashParamCheck(
    paramCombo = list(
      dimToken = 192,
      numHeads = 8,
      positionalEncoding = list(name = "StochasticConvPE")
    ),
    temporal = TRUE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_true(outStoch$ok)

  outAlibi <- flashParamCheck(
    paramCombo = list(
      dimToken = 192,
      numHeads = 8,
      positionalEncoding = list(name = "ALiBiPE")
    ),
    temporal = TRUE,
    temporalSettings = list(),
    envInfo = envInfo
  )
  expect_false(outAlibi$ok)
  expect_true(any(grepl("not supported by flash-attn v2", outAlibi$reasons)))
})

test_that("flashParamCheck picks PE from temporalSettings when paramCombo lacks it", {
  envInfo <- list(info = list(deviceName = "NVIDIA A100-SXM4-40GB"))

  out <- flashParamCheck(
    paramCombo = list(dimToken = 192, numHeads = 8),
    temporal = TRUE,
    temporalSettings = list(positionalEncoding = list(name = "RelativePE")),
    envInfo = envInfo
  )
  expect_false(out$ok)
  expect_true(any(grepl("RelativePE", paste(out$reasons, collapse = " "))))
})

test_that("flashEnvCheck flags non-Linux OS", {
  fake <- makeFakeTorch(
    cudaAvailable = TRUE,
    cudaVersion = "12.1",
    capMajor = 8L, capMinor = 0L,
    deviceName = "NVIDIA A100-SXM4-40GB",
    bf16Supported = TRUE
  )
  localMockFlashBindings(fakeTorch = fake, flashModuleAvailable = TRUE, .scope = environment())
  testthat::local_mocked_bindings(
    getSysName = function() "windows",
    .package = "DeepPatientLevelPrediction",
    .env = environment()
  )

  env <- DeepPatientLevelPrediction:::flashEnvCheck(
    estimatorSettings = list(device = "cuda", precision = "bfloat16")
  )
  expect_false(env$ok)
  expect_true(any(grepl("requires Linux; detected", env$reasons)))
})

test_that("flashParamCheck flags unknown positionalEncoding", {
  envInfo <- list(info = list(deviceName = "NVIDIA A100-SXM4-40GB"))
  out <- DeepPatientLevelPrediction:::flashParamCheck(
    paramCombo = list(dimToken = 192, numHeads = 8),
    temporal = TRUE,
    temporalSettings = list(positionalEncoding = list(name = "CustomPE")), # unknown
    envInfo = envInfo
  )
  expect_false(out$ok)
  expect_true(any(grepl("Unknown positionalEncoding 'CustomPE'", out$reasons)))
})

test_that("formatFlashErrors returns empty string when reasons is empty", {
  msg <- DeepPatientLevelPrediction:::formatFlashErrors("Header", character(0))
  expect_identical(msg, "")
})
