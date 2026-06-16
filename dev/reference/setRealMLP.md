# Create RealMLP Settings

Create settings for the RealMLP model (binary classification).
Preprocessing (robust scaling + clipping) is assumed to be handled
upstream.

## Usage

``` r
setRealMLP(
  numLayers = 3L,
  sizeHidden = 256L,
  dropout = 0.15,
  sizeEmbedding = 64L,
  labelSmoothing = 0,
  numericEmbeddingMode = "scale",
  numericNumFrequencies = 8L,
  numericPeriodicInitStd = 0.1,
  numericPbldHiddenDim = 16L,
  numericPbldEmbeddingDim = 4L,
  dataDependentInitMode = "paper_lsuv",
  dataDependentInitTargetVar = 1,
  dataDependentInitMaxRows = 65536L,
  dataDependentInitGainClip = 10,
  dataDependentInitBiasRefitSteps = 2L,
  scalingLrMult = 6,
  biasLrMult = 0.1,
  actLrMult = 0.1,
  embeddingLrMult = 0.1,
  paperMode = TRUE,
  tokenAggregation = "auto",
  featureScaleMode = "auto",
  device = "cpu",
  hyperParamSearch = "grid",
  randomSample = 100,
  randomSampleSeed = NULL
)
```

## Arguments

- numLayers:

  hidden layers (default 3)

- sizeHidden:

  hidden width (default 256)

- dropout:

  base dropout p (default 0.15; scheduled with flat_cos)

- sizeEmbedding:

  embedding dim for compatibility with existing Embedding (default 64)

- labelSmoothing:

  epsilon for label smoothing (default 0.0 for AUROC mode)

- numericEmbeddingMode:

  numeric token embedding mode: "scale", "pl", or "pbld"

- numericNumFrequencies:

  periodic frequency count for PL/PBLD modes

- numericPeriodicInitStd:

  std used to initialize periodic frequencies

- numericPbldHiddenDim:

  hidden width for PBLD per-feature block

- numericPbldEmbeddingDim:

  low-dimensional PBLD output width before projection

- dataDependentInitMode:

  data-dependent init mode: "paper_lsuv" or "current"

- dataDependentInitTargetVar:

  target pre-activation variance per neuron

- dataDependentInitMaxRows:

  max sampled rows for init statistics (0 means all sampled rows)

- dataDependentInitGainClip:

  optional clip on LSUV gain multipliers (\>1 enables clipping)

- dataDependentInitBiasRefitSteps:

  bias recenter iterations during data-dependent init

- scalingLrMult:

  LR multiplier for scaling parameters (default 6.0)

- biasLrMult:

  LR multiplier for bias parameters (default 0.1)

- actLrMult:

  LR multiplier for parametric activation parameters (default 0.1)

- embeddingLrMult:

  LR multiplier for embedding parameters (default 0.1)

- paperMode:

  if TRUE, enforce paper-aligned defaults where possible

- tokenAggregation:

  token aggregation mode: "auto", "mean", "sum", "sum_len_norm"

- featureScaleMode:

  feature scale mode: "auto", "scalar", "vector"

- device:

  "cpu" or "cuda" (default "cpu")

- hyperParamSearch:

  Which kind of hyperparameter search to use: random sampling or
  exhaustive grid search. Default: "grid"

- randomSample:

  How many random samples from hyperparameter space to use when
  \`hyperParamSearch = "random"\`

- randomSampleSeed:

  Random seed to sample hyperparameter combinations
