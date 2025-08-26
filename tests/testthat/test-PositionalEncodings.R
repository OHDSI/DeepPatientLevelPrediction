
path <- system.file("python", package = "DeepPatientLevelPrediction")
createTestHarness <- function(peModule,
                              mockFeatureInfo,
                              batchSize = 4L,
                              seqLen = 16L, 
                              dimToken = 32L,
                              numHeads = 4L,
                              onlyClassToken = FALSE) {
  transformer <- reticulate::import_from_path("Transformer", path = path)$Transformer
  transformerBlock <- reticulate::import_from_path("Transformer", path = path)$TransformerBlock
  
  fullTransformer <- transformer(
    feature_info = mockFeatureInfo,
    num_blocks = 1L,
    dim_token = as.integer(dimToken),
    num_heads = as.integer(numHeads),
    att_dropout = 0.0,
    ffn_dropout = 0.0,
    dim_hidden = as.integer(dimToken * 4),
    pe_module = peModule
  )

  block <- transformerBlock(
    dim_token = as.integer(dimToken),
    num_heads = as.integer(numHeads),
    att_dropout = 0.0,
    ffn_dropout = 0.0,
    dim_hidden = as.integer(dimToken * 4),
    pe_module = peModule,
    only_class_token = onlyClassToken
  )
  
  # Create deterministic input data
  torch$manual_seed(42)
  vocabSize <- mockFeatureInfo$get_vocabulary_size()
  featureIds <- torch$randint(low = 1L, high = vocabSize, size = c(batchSize, seqLen), dtype = torch$long)
  featureValues <- torch$randn(batchSize, seqLen)
  timeIds <- torch$arange(0L, (seqLen + 1) * 5L, 5L)$unsqueeze(0L)$expand(batchSize, -1L)$long()
  sequenceLengths <- torch$tensor(rep(as.integer(seqLen), batchSize), dtype = torch$long)
  
  xSeqLen <- if (onlyClassToken) seqLen + 1L else seqLen
  xBlock <- torch$randn(batchSize, xSeqLen, dimToken)
  maskBlock <- torch$ones(batchSize, xSeqLen, dtype = torch$bool)
  timeIdsBlock <- torch$arange(0L, xSeqLen * 5L, 5L)$unsqueeze(0L)$expand(batchSize, -1L)$long()
  inputData <- list(
    feature_ids = featureIds,
    feature_values = featureValues,
    time_ids = timeIds,
    sequence_lengths = sequenceLengths
  )
  return(list(block = block, fullModel = fullTransformer, inputData = inputData,
              x = xBlock, mask = maskBlock, timeIds = timeIdsBlock))
}

featureInfoMock <- reticulate::py_run_string("
import torch
class FeatureInfoMock:
    def __init__(self, max_time_id, vocab_size=10, numerical_ids=None):
        self._max_time_id = int(max_time_id)
        self._vocab_size = int(vocab_size)
        
        # Default numerical IDs are simple but sufficient to test both code paths.
        if numerical_ids is None:
            self._numerical_ids = torch.tensor([1, 2], dtype=torch.long)
        else:
            self._numerical_ids = torch.tensor(numerical_ids, dtype=torch.long)

    def get_max_time_id(self):
        return self._max_time_id

    def get_vocabulary_size(self):
        return self._vocab_size
    
    def get_numerical_feature_ids(self):
        return self._numerical_ids
", convert = FALSE)

test_that("NoPositionalEncoding runs and is the baseline", {
  mockFeatureInfo <- featureInfoMock$FeatureInfoMock(max_time_id = 512L)
  peFactory <- reticulate::import_from_path("PositionalEncodings", path = path)$NoPositionalEncoding
  peModule <- peFactory()
  
  harness <- createTestHarness(peModule, mockFeatureInfo)
  
  output <- harness$block(harness$x, harness$mask, harness$timeIds)
  
  expect_equal(output$shape[0], 4L)
  expect_equal(output$shape[1], 16L)
  expect_equal(output$shape[2], 32L)
})

test_that("Additive PEs (Sinusoidal, Learnable, TapePE) run and modify output", {
  peClasses <- c("SinusoidalPE", "LearnablePE", "TapePE")
  positionalEncodings <- reticulate::import_from_path("PositionalEncodings", path = path)
  createPeModule <- positionalEncodings$create_positional_encoding_module

  mockFeatureInfo <- featureInfoMock$FeatureInfoMock(max_time_id = 512L)
  
  noPeConfig <- list(name = "NoPositionalEncoding")
  noPeModelParams <- list(
    dim_token = 32L, num_heads = 4L, feature_info = mockFeatureInfo,
    positional_encoding = noPeConfig
  )
  noPeModule <- createPeModule(noPeModelParams)
  harnessNone <- createTestHarness(positionalEncodings$NoPositionalEncoding(), mockFeatureInfo)
  outputNone <- harnessNone$fullModel(harnessNone$inputData)

  for (peName in peClasses) {
    peConfig <- list(name = peName)
    modelParams <- list(
      dim_token = 32L, num_heads = 4L, feature_info = mockFeatureInfo,
      positional_encoding = peConfig
    )
    peModule <- createPeModule(modelParams)
    harness <- createTestHarness(peModule, mockFeatureInfo)
    output <- harness$fullModel(harness$inputData)

    expect_false(torch$allclose(output, outputNone))
    expect_equal(output$shape[0], 4L)
  }
})

test_that("Standard Attention PEs modify TransformerBlock output", {
  # This group includes ALL PEs that work with the default MultiHeadAttention
  peClasses <- c("RotaryPE", "RelativePE", "TemporalPE", "StochasticConvPE", "HybridRoPEConvPE", "ALiBiPE")
  positionalEncodings <- reticulate::import_from_path("PositionalEncodings", path = path)
  createPeModule <- positionalEncodings$create_positional_encoding_module
  
  mockFeatureInfo <- featureInfoMock$FeatureInfoMock(max_time_id = 512L)
  
  harnessNone <- createTestHarness(positionalEncodings$NoPositionalEncoding(), mockFeatureInfo)
  outputNone <- harnessNone$block(harnessNone$x, harnessNone$mask, harnessNone$timeIds)

  for (peName in peClasses) {
    message("Testing Standard Attention PE: ", peName)
    
    peConfig <- list(name = peName)
    if (peName == "TemporalPE") {
      peConfig$abs_config <- list(name = "SinusoidalPE")
      peConfig$rel_config <- list(name = "RelativePE")
    }
    
    modelParams <- reticulate::r_to_py(list(
      dim_token = 32L, num_heads = 4L, feature_info = mockFeatureInfo,
      positional_encoding = peConfig
    ))
    
    peModule <- createPeModule(modelParams)
    harness <- createTestHarness(peModule, mockFeatureInfo)
    
    # Verify that the standard MultiHeadAttention was created
    attnName <- reticulate::py_to_r(harness$block$attn$`__class__`$`__name__`)
    expect_equal(attnName, "MultiHeadAttention")
    
    output <- harness$block(harness$x, harness$mask, harness$timeIds)
    expect_false(torch$allclose(output, outputNone))
    expect_equal(output$shape[0], 4L)
    expect_equal(output$shape[1], 16L)
    expect_equal(output$shape[2], 32L)
  }
})

test_that("Custom Attention PEs instantiate correctly and modify output", {
  # This group includes ALL PEs that provide their own attention class
  peConfigMap <- list(
    "EfficientRPE" = list(name = "EfficientRPE", expected_attn = "eRPEAttention"),
    "TUPE" = list(name = "TUPE", base_pe_config = list(name = "SinusoidalPE"), expected_attn = "TUPEMultiHeadAttention")
  )
  
  positionalEncodings <- reticulate::import_from_path("PositionalEncodings", path = path)
  createPeModule <- positionalEncodings$create_positional_encoding_module
  
  mockFeatureInfo <- featureInfoMock$FeatureInfoMock(max_time_id = 512L)
  
  harnessNone <- createTestHarness(positionalEncodings$NoPositionalEncoding(), mockFeatureInfo)
  outputNone <- harnessNone$block(harnessNone$x, harnessNone$mask, harnessNone$timeIds)
  
  for (peName in names(peConfigMap)) {
    peConfig <- peConfigMap[[peName]]
    expectedAttnName <- peConfig$expected_attn
    peConfig$expected_attn <- NULL # Remove from config before passing to factory
    
    modelParams <- reticulate::r_to_py(list(
      dim_token = 32L, num_heads = 4L, feature_info = mockFeatureInfo,
      positional_encoding = peConfig
    ))
    
    peModule <- createPeModule(modelParams)
    harness <- createTestHarness(peModule, mockFeatureInfo)
    
    actualAttnName <- reticulate::py_to_r(harness$block$attn$`__class__`$`__name__`)
    expect_equal(actualAttnName, expectedAttnName)
    
    output <- harness$block(harness$x, harness$mask, harness$timeIds)
    expect_false(torch$allclose(output, outputNone))
    expect_equal(output$shape[0], 4L)
    expect_equal(output$shape[1], 16L)
    expect_equal(output$shape[2], 32L)

  }
})

test_that("All PE modules handle asymmetric CLS token attention", {
  peClasses <- c(
    "NoPositionalEncoding", "SinusoidalPE", "LearnablePE", "TapePE", 
    "RotaryPE", "RelativePE", "EfficientRPE", "StochasticConvPE",
    "TUPE", "TemporalPE", "HybridRoPEConvPE", "ALiBiPE"
  )
  
  positionalEncodings <- reticulate::import_from_path("PositionalEncodings", path = path)
  createPeModule <- positionalEncodings$create_positional_encoding_module
  
  mockFeatureInfo <- featureInfoMock$FeatureInfoMock(max_time_id = 512L)
  
  for (peName in peClasses) {
    peConfig <- list(name = peName)
    
    if (peName == "TUPE") {
      peConfig$base_pe_config <- list(name = "SinusoidalPE")
    }
    if (peName == "TemporalPE") {
      peConfig$abs_config <- list(name = "SinusoidalPE")
      peConfig$rel_config <- list(name = "RelativePE")
    }
    
    modelParams <- reticulate::r_to_py(list(
      dim_token = 32L,
      num_heads = 4L,
      feature_info = mockFeatureInfo,
      positional_encoding = peConfig
    ))
    
    peModule <- createPeModule(modelParams)
    
    
    harness <- createTestHarness(
      peModule = peModule,
      mockFeatureInfo = mockFeatureInfo,
      onlyClassToken = TRUE,
      seqLen = 16L 
    )
    
    expect_no_error({
      output <- harness$block(harness$x, harness$mask, harness$timeIds)
    }, message = paste("PE method", peName, "failed on asymmetric CLS token attention."))
    
    expect_equal(output$shape[0], 4L)
    expect_equal(output$shape[1], 1L)
    expect_equal(output$shape[2], 32L)
  }
})
