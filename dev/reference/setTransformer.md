# create settings for training a transformer

A transformer model

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

  number of transformer blocks

- dimToken:

  dimension of each token (embedding size)

- dimOut:

  dimension of output, usually 1 for binary problems

- numHeads:

  number of attention heads

- attDropout:

  dropout to use on attentions

- ffnDropout:

  dropout to use in feedforward block

- dimHidden:

  dimension of the feedworward block

- dimHiddenRatio:

  dimension of the feedforward block as a ratio of dimToken (embedding
  size)

- temporal:

  Whether to use a transformer with temporal data

- temporalSettings:

  settings for the temporal transformer. Which include -
  \`positionalEncoding\`: Positional encoding to use, either a character
  or a list with name and settings, default 'SinusoidalPE' with dropout
  0.1 - \`maxSequenceLength\`: Maximum sequence length, sequences longer
  than This will be truncated and/or padded to this length either a
  number or 'max' for the Maximum - \`truncation\`: Truncation method,
  only 'tail' is supported - \`timeTokens\`: Whether to use time tokens,
  default TRUE

- estimatorSettings:

  created with \`setEstimator\`

- hyperParamSearch:

  what kind of hyperparameter search to do, default 'random'

- randomSample:

  How many samples to use in hyperparameter search if random

- randomSampleSeed:

  Random seed to sample hyperparameter combinations

## Value

list of settings for the transformer model

## Details

The non-temporal transformer is from https://arxiv.org/abs/2106.11959
