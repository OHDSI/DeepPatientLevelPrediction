library(torch)
nSubjects <- 75e3
nFeatures <- 1e3
torch::torch_manual_seed(seed=42)

rowIds <- torch::torch_randint(1,nSubjects, c(1e6,1))
columnIds <- torch::torch_randint(1,nFeatures, c(1e6, 1))

tensor <- torch::torch_cat(c(rowIds, columnIds), dim=2)

sortedTensor <- tensor$sort(dim=1)[[1]]

counts <- as.integer(torch::torch_unique_consecutive(sortedTensor[,1], return_counts=TRUE)[[3]])

microbenchmark::microbenchmark(splitted <- torch::torch_split(sortedTensor[,2], counts), times=1)
