# Create RealMLP Settings

Creates model and hyperparameter-search settings for RealMLP binary
classification.

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
  estimatorSettings = setEstimator(learningRate = 0.002, weightDecay = 0.02, batchSize =
    256L, epochs = 256L, scheduler = NULL, earlyStopping = NULL, metric = "loss"),
  hyperParamSearch = "grid",
  randomSample = 100,
  randomSampleSeed = NULL,
  device = NULL
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

- estimatorSettings:

  Estimator settings created by
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/dev/reference/setEstimator.md).
  The default trains as in the RealMLP paper: AdamW with learning rate
  2e-3 and weight decay 0.02, batches of 256 and 256 epochs, keeping the
  epoch with the lowest validation loss. RealMLP follows its own
  learning-rate schedule, so a `scheduler` is ignored with a warning,
  and its own loss, so `criterion` is not used. Early stopping is
  possible, but the paper trains for all epochs.

- hyperParamSearch:

  Which kind of hyperparameter search to use: random sampling or
  exhaustive grid search. Default: "grid"

- randomSample:

  How many random samples from hyperparameter space to use when
  `hyperParamSearch = "random"`

- randomSampleSeed:

  Random seed to sample hyperparameter combinations

- device:

  Deprecated. Use `estimatorSettings = setEstimator(device = )`.

## Value

A `modelSettings` object for use with `PatientLevelPrediction`.

## Details

The implementation is based on [Holzmuller et al.
(2024)](https://arxiv.org/abs/2407.04491). Preprocessing, including
robust scaling and clipping, is handled upstream.

## Examples

``` r
realMlpSettings <- setRealMLP(
  sizeHidden = c(128, 256),
  dropout = c(0.1, 0.2),
  hyperParamSearch = "random",
  randomSample = 2,
  randomSampleSeed = 42
)
length(realMlpSettings$param)
#> [1] 2
```
