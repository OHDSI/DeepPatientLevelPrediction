library(torch)
source('./extras/ResNet.R')
source('./extras/Estimator.R')

torch::torch_manual_seed(42)
features <- 500
rows <-11250

data <- torch::torch_randint(0, features, size=c(rows, 200), dtype=torch::torch_long())
targets <- torch::torch_randint(0, 2, size=c(rows), dtype=torch::torch_float32())


modelParams <- list('numLayers'=2L,
                    'sizeHidden'=256,
                    'hiddenFactor'=1L,
                    'residualDropout'=0.0,
                    'hiddenDropout'=0.0,
                    'sizeEmbedding'=128,
                    'catFeatures'=features)

fitParams <- list('epochs'=20,
                  'learningRate'=3e-4,
                  'weightDecay'=0,
                  'batchSize'=2056,
                  'posWeight'=1)


for (i in 1:100) {
  estimator <- Estimator$new(baseModel = ResNet,
                             modelParameters = modelParams,
                             fitParameters = fitParams,
                             device='cuda:0')
  dataset <- torch::tensor_dataset(data, targets)
  
  estimator$fit(dataset)
  torch::cuda_empty_cache()
}
