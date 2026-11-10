# TrainingCache

Stores hyperparameter-search progress so interrupted model training can
be resumed from an analysis directory.

## Value

An R6 class generator for persistent training caches.

## Methods

### Public methods

- [`TrainingCache$new()`](#method-TrainingCache-initialize)

- [`TrainingCache$isParamGridIdentical()`](#method-TrainingCache-isParamGridIdentical)

- [`TrainingCache$isFingerprintIdentical()`](#method-TrainingCache-isFingerprintIdentical)

- [`TrainingCache$saveFingerprint()`](#method-TrainingCache-saveFingerprint)

- [`TrainingCache$saveGridSearchPredictions()`](#method-TrainingCache-saveGridSearchPredictions)

- [`TrainingCache$saveModelParams()`](#method-TrainingCache-saveModelParams)

- [`TrainingCache$getGridSearchPredictions()`](#method-TrainingCache-getGridSearchPredictions)

- [`TrainingCache$isFull()`](#method-TrainingCache-isFull)

- [`TrainingCache$getLastGridSearchIndex()`](#method-TrainingCache-getLastGridSearchIndex)

- [`TrainingCache$dropCache()`](#method-TrainingCache-dropCache)

- [`TrainingCache$trimPerformance()`](#method-TrainingCache-trimPerformance)

- [`TrainingCache$clone()`](#method-TrainingCache-clone)

------------------------------------------------------------------------

### `TrainingCache$new()`

Creates a new training cache

#### Usage

    TrainingCache$new(inDir)

#### Arguments

- `inDir`:

  Path to the analysis directory

------------------------------------------------------------------------

### `TrainingCache$isParamGridIdentical()`

Checks whether the parameter grid in the model settings is identical to
the cached parameters.

#### Usage

    TrainingCache$isParamGridIdentical(inModelParams)

#### Arguments

- `inModelParams`:

  Parameter grid from the model settings

#### Returns

Whether the provided and cached parameter grid is identical

------------------------------------------------------------------------

### `TrainingCache$isFingerprintIdentical()`

Checks whether the cached results were produced with the same data and
training configuration.

#### Usage

    TrainingCache$isFingerprintIdentical(inFingerprint)

#### Arguments

- `inFingerprint`:

  Fingerprint of the current data and configuration

#### Returns

Whether the provided and cached fingerprint are identical

------------------------------------------------------------------------

### `TrainingCache$saveFingerprint()`

Saves the fingerprint of the data and configuration to the training
cache

#### Usage

    TrainingCache$saveFingerprint(inFingerprint)

#### Arguments

- `inFingerprint`:

  Fingerprint of the current data and configuration

------------------------------------------------------------------------

### `TrainingCache$saveGridSearchPredictions()`

Saves the grid search results to the training cache

#### Usage

    TrainingCache$saveGridSearchPredictions(inGridSearchPredictions)

#### Arguments

- `inGridSearchPredictions`:

  Grid search predictions

------------------------------------------------------------------------

### `TrainingCache$saveModelParams()`

Saves the parameter grid to the training cache

#### Usage

    TrainingCache$saveModelParams(inModelParams)

#### Arguments

- `inModelParams`:

  Parameter grid from the model settings

------------------------------------------------------------------------

### `TrainingCache$getGridSearchPredictions()`

Gets the grid search results from the training cache

#### Usage

    TrainingCache$getGridSearchPredictions()

#### Returns

Grid search results from the training cache

------------------------------------------------------------------------

### `TrainingCache$isFull()`

Check if cache is full

#### Usage

    TrainingCache$isFull()

#### Returns

A logical value.

------------------------------------------------------------------------

### `TrainingCache$getLastGridSearchIndex()`

Gets the last index from the cached grid search

#### Usage

    TrainingCache$getLastGridSearchIndex()

#### Returns

Last grid search index

------------------------------------------------------------------------

### `TrainingCache$dropCache()`

Remove the training cache from the analysis path

#### Usage

    TrainingCache$dropCache()

------------------------------------------------------------------------

### `TrainingCache$trimPerformance()`

Trims the performance of the hyperparameter results by removing the
predictions from all but the best performing hyperparameter

#### Usage

    TrainingCache$trimPerformance(hyperparameterResults, maximize = TRUE)

#### Arguments

- `hyperparameterResults`:

  List of hyperparameter results

- `maximize`:

  Whether higher performance values are better

------------------------------------------------------------------------

### `TrainingCache$clone()`

The objects of this class are cloneable with this method.

#### Usage

    TrainingCache$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
cacheDirectory <- tempfile("training-cache-")
dir.create(cacheDirectory)
cache <- trainingCache$new(cacheDirectory)
cache$saveModelParams(list(list(sizeHidden = 64)))
cache$isParamGridIdentical(list(list(sizeHidden = 64)))
#> [1] TRUE
unlink(cacheDirectory, recursive = TRUE)
```
