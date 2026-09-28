# Create Transformer Settings

Creates model and hyperparameter-search settings for either a
non-temporal or temporal transformer.

## Usage

``` r
setTransformer(
  numBlocks = 3,
  dimToken = 192,
  dimOut = 1,
  numHeads = 8,
  attDropout = 0.2,
  ffnDropout = 0.1,
  dimHidden = 256,
  dimHiddenRatio = NULL,
  temporal = FALSE,
  temporalSettings = list(positionalEncoding = list(name = "SinusoidalPE", dropout =
    0.1), maxSequenceLength = 256, truncation = "tail", timeTokens = TRUE),
  estimatorSettings = setEstimator(weightDecay = 1e-06, batchSize = 1024, epochs = 10,
    seed = NULL),
  hyperParamSearch = "random",
  randomSample = 1,
  randomSampleSeed = NULL
)
```

## Arguments

- numBlocks:

  Number of transformer blocks.

- dimToken:

  Token dimension, which is also the embedding dimension.

- dimOut:

  Output dimension, usually one for binary prediction.

- numHeads:

  Number of attention heads.

- attDropout:

  Attention-dropout probability.

- ffnDropout:

  Feed-forward-network dropout probability.

- dimHidden:

  Hidden dimension of the feed-forward network.

- dimHiddenRatio:

  Feed-forward hidden dimension as a ratio of `dimToken`. Exactly one of
  `dimHidden` and `dimHiddenRatio` must be `NULL`.

- temporal:

  Whether to configure a transformer for temporal covariates.

- temporalSettings:

  A list with `positionalEncoding`, `maxSequenceLength`, `truncation`,
  and `timeTokens`. The only supported truncation strategy is `"tail"`.

- estimatorSettings:

  Estimator settings created by
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/reference/setEstimator.md).

- hyperParamSearch:

  Hyperparameter-search strategy, either `"random"` or `"grid"`.

- randomSample:

  Number of combinations sampled when `hyperParamSearch = "random"`.

- randomSampleSeed:

  Random seed used when sampling combinations.

## Value

A `modelSettings` object. Temporal settings are stored as attributes on
its parameter grid.

## Details

The non-temporal architecture is based on [Gorishniy et al.
(2021)](https://arxiv.org/abs/2106.11959). For temporal data, positional
encoding can be configured through `temporalSettings`.

## Examples

``` r
transformerSettings <- setTransformer(
  numBlocks = 2,
  dimToken = 64,
  numHeads = 4,
  dimHidden = 128
)

temporalSettings <- setTransformer(
  numBlocks = 1,
  dimToken = 32,
  numHeads = 4,
  dimHidden = 64,
  temporal = TRUE,
  temporalSettings = list(
    positionalEncoding = "SinusoidalPE",
    maxSequenceLength = 128,
    truncation = "tail",
    timeTokens = TRUE
  )
)
```
