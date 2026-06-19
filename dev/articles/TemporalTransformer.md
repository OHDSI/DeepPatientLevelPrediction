# Building Temporal Transformer Models

## Introduction

This vignette shows how to build a temporal transformer model with
`DeepPatientLevelPrediction`. It assumes you are already familiar with
building a regular `PatientLevelPrediction` analysis and have read the
first model vignette.

The temporal transformer uses sequence data instead of the usual
two-dimensional patient-by-feature matrix. Each covariate record must
include a `timeId` so the model can receive feature IDs, feature values,
and the time step for each feature. The temporal sequence is created
when you extract `plpData`.

## Create Temporal Data

Temporal transformer models need `plpData` built with temporal sequence
covariate settings. The example below extracts demographic and condition
features over the one year window before index.

``` r

temporalCovariateSettings <-
  FeatureExtraction::createTemporalSequenceCovariateSettings(
    useDemographicsAge = TRUE,
    useDemographicsGender = TRUE,
    useConditionOccurrence = TRUE,
    sequenceStartDay = -365,
    sequenceEndDay = -1
  )

temporalPlpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  restrictPlpDataSettings =
    PatientLevelPrediction::createRestrictPlpDataSettings(),
  covariateSettings = temporalCovariateSettings
)
```

Use the same `databaseDetails`, target and outcome cohort definitions,
and `populationSettings` that you would use for a regular PLP analysis.

## Configure The Model

Use
[`setTransformer()`](https://ohdsi.github.com/DeepPatientLevelPrediction/dev/reference/setTransformer.md)
with `temporal = TRUE` to enable the temporal path. Temporal behavior is
controlled through `temporalSettings`.

The most important settings are:

- `maxSequenceLength`: the maximum number of time-ordered feature
  records per person. Use an integer for a fixed length or `"max"` to
  use the maximum sequence length in the data.
- `truncation`: how to truncate sequences longer than
  `maxSequenceLength`. Currently only `"tail"` is supported.
- `timeTokens`: whether to include explicit time tokens in the model
  input.
- `positionalEncoding`: the positional encoding used by the transformer.
  This can be a character value such as `"SinusoidalPE"` or a list with
  the encoding name and settings.

For reproducible model definitions, set `temporalSettings` explicitly
instead of relying on defaults.

``` r

modelSettings <- DeepPatientLevelPrediction::setTransformer(
  numBlocks = 1L,
  dimToken = 8L,
  dimOut = 1L,
  numHeads = 2L,
  attDropout = 0.0,
  ffnDropout = 0.2,
  dimHidden = 32L,
  temporal = TRUE,
  temporalSettings = list(
    positionalEncoding = list(
      name = "SinusoidalPE",
      dropout = 0.1
    ),
    maxSequenceLength = 256L,
    truncation = "tail",
    timeTokens = FALSE
  ),
  estimatorSettings = DeepPatientLevelPrediction::setEstimator(
    learningRate = 3e-4,
    weightDecay = 1e-6,
    batchSize = 64L,
    epochs = 3L,
    device = "cpu"
  ),
  randomSample = 1L
)
```

Use `device = "cuda"` or a specific CUDA device such as `"cuda:0"` when
you have an NVIDIA GPU available.

## Run The Analysis

The temporal transformer is passed to
[`PatientLevelPrediction::runPlp()`](https://ohdsi.github.io/PatientLevelPrediction/reference/runPlp.html)
in the same way as other DeepPLP model settings.

``` r

temporalTransformerResult <- PatientLevelPrediction::runPlp(
  plpData = temporalPlpData,
  outcomeId = 3,
  modelSettings = modelSettings,
  analysisId = "TemporalTransformer",
  analysisName = "Testing temporal transformer",
  populationSettings = populationSettings,
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting(
    splitSeed = 42
  ),
  preprocessSettings = PatientLevelPrediction::createPreprocessSettings(),
  executeSettings = PatientLevelPrediction::createExecuteSettings(
    runSplitData = TRUE,
    runSampleData = FALSE,
    runFeatureEngineering = FALSE,
    runPreprocessData = TRUE,
    runModelDevelopment = TRUE,
    runCovariateSummary = FALSE
  ),
  saveDirectory = file.path(getwd(), "TemporalTransformer")
)
```

## Practical Notes

Temporal transformers are usually more expensive than non-temporal
models because attention scales with sequence length. Start with a small
`maxSequenceLength`, a small number of blocks, and a CPU-compatible test
run. Increase model size and move to CUDA only after the data extraction
and model wiring are working.

The order and density of the extracted temporal features matter. If the
model is too slow or uses too much memory, reduce the number of temporal
covariates in `FeatureExtraction`, shorten the time window, or lower
`maxSequenceLength`.

## Acknowledgments

Considerable work has been dedicated to provide the
`DeepPatientLevelPrediction` package.

``` r

citation("DeepPatientLevelPrediction")
```

    ## To cite package 'DeepPatientLevelPrediction' in publications use:
    ## 
    ##   Fridgeirsson E, Reps J, Chan You S, Kim C, John H (2026).
    ##   _DeepPatientLevelPrediction: Deep Learning for Patient Level
    ##   Prediction Using Data in the OMOP Common Data Model_. R package
    ##   version 2.3.0.9999,
    ##   <https://github.com/OHDSI/DeepPatientLevelPrediction>.
    ## 
    ## A BibTeX entry for LaTeX users is
    ## 
    ##   @Manual{,
    ##     title = {DeepPatientLevelPrediction: Deep Learning for Patient Level Prediction Using Data in the
    ## OMOP Common Data Model},
    ##     author = {Egill Fridgeirsson and Jenna Reps and Seng {Chan You} and Chungsoo Kim and Henrik John},
    ##     year = {2026},
    ##     note = {R package version 2.3.0.9999},
    ##     url = {https://github.com/OHDSI/DeepPatientLevelPrediction},
    ##   }

**Please reference this paper if you use the PLP Package in your work:**

[Reps JM, Schuemie MJ, Suchard MA, Ryan PB, Rijnbeek PR. Design and
implementation of a standardized framework to generate and evaluate
patient-level prediction models using observational healthcare data. J
Am Med Inform Assoc.
2018;25(8):969-975.](http://dx.doi.org/10.1093/jamia/ocy032)
