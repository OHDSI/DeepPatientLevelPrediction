# Create Model Settings with Custom Embeddings

Configures a model to use supplied embeddings, such as Poincare
embeddings or embeddings from a foundation model.

## Usage

``` r
setCustomEmbeddingModel(
  embeddingFilePath,
  modelSettings = setTransformer(numBlocks = 3, dimToken = 16, dimOut = 1, numHeads = 4,
    attDropout = 0.2, ffnDropout = 0.1, dimHidden = 32, estimatorSettings =
    setEstimator(learningRate = "auto", weightDecay = 1e-04, batchSize = 256, epochs = 2,
    seed = NULL, device = "cpu"), hyperParamSearch = "random", randomSample = 1),
  embeddingsClass = "CustomEmbeddings",
  curvature = 1
)
```

## Arguments

- embeddingFilePath:

  Path to a PyTorch file containing a dictionary with `concept_ids`, a
  PyTorch long tensor, and `embeddings`, a PyTorch float tensor.

- modelSettings:

  Settings for a model with an embedding layer named `embedding`. The
  supplied embeddings replace that layer.

- embeddingsClass:

  Embedding implementation, either `"CustomEmbeddings"` or
  `"PoincareEmbeddings"`.

- curvature:

  For `"PoincareEmbeddings"`, the magnitude `c` of the curvature `-c` of
  the Poincare ball that the embeddings were trained in. The norm of
  every embedding must be below `1 / sqrt(c)`. This is the curvature
  itself, not the parameter of a library that transforms it: HypLL, used
  by OHDSI/context, applies softplus, so an untrained curvature
  parameter of 1 there means `c = log(1 + exp(1))`, about 1.313. If the
  curvature was trained, use its final value after softplus.

## Value

A `modelSettings` object configured to initialize custom embeddings.

## Examples

``` r
if (FALSE) { # \dontrun{
# Requires Python and the package's Python dependencies, including PyTorch.
# See vignette("Installing") for setup instructions.
embeddingFilePath <- tempfile(fileext = ".pt")

# Toy embeddings: one row per concept ID, with eight values per row.
conceptIds <- c(1L, 2L, 3L)
embeddings <- list(
  concept_ids = torch$tensor(conceptIds, dtype = torch$long),
  embeddings = torch$randn(length(conceptIds), 8L, dtype = torch$float)
)
torch$save(embeddings, embeddingFilePath)

modelSettings <- setCustomEmbeddingModel(
  embeddingFilePath = embeddingFilePath,
  modelSettings = setDefaultTransformer()
)

# Clean up this example's file. In real use, keep it for as long as you use
# the model: modelSettings and the saved model store its path, not the
# embedding tensors, and predicting with a saved model reads it again.
unlink(embeddingFilePath)
} # }
```
