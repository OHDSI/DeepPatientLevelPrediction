# TrainingCache

Parameter caching for training persistence and continuity

## Value

Whether the provided and cached parameter grid is identical

Grid search results from the training cache

Boolen

Last grid search index

## Methods

### Public methods

- [`trainingCache$new()`](#method-TrainingCache-new)

- [`trainingCache$isParamGridIdentical()`](#method-TrainingCache-isParamGridIdentical)

- [`trainingCache$saveGridSearchPredictions()`](#method-TrainingCache-saveGridSearchPredictions)

- [`trainingCache$saveModelParams()`](#method-TrainingCache-saveModelParams)

- [`trainingCache$getGridSearchPredictions()`](#method-TrainingCache-getGridSearchPredictions)

- [`trainingCache$isFull()`](#method-TrainingCache-isFull)

- [`trainingCache$getLastGridSearchIndex()`](#method-TrainingCache-getLastGridSearchIndex)

- [`trainingCache$dropCache()`](#method-TrainingCache-dropCache)

- [`trainingCache$trimPerformance()`](#method-TrainingCache-trimPerformance)

- [`trainingCache$clone()`](#method-TrainingCache-clone)

------------------------------------------------------------------------

### Method `new()`

Creates a new training cache

#### Usage

    trainingCache$new(inDir)

#### Arguments

- `inDir`:

  Path to the analysis directory

------------------------------------------------------------------------

### Method `isParamGridIdentical()`

Checks whether the parameter grid in the model settings is identical to
the cached parameters.

#### Usage

    trainingCache$isParamGridIdentical(inModelParams)

#### Arguments

- `inModelParams`:

  Parameter grid from the model settings

------------------------------------------------------------------------

### Method `saveGridSearchPredictions()`

Saves the grid search results to the training cache

#### Usage

    trainingCache$saveGridSearchPredictions(inGridSearchPredictions)

#### Arguments

- `inGridSearchPredictions`:

  Grid search predictions

------------------------------------------------------------------------

### Method `saveModelParams()`

Saves the parameter grid to the training cache

#### Usage

    trainingCache$saveModelParams(inModelParams)

#### Arguments

- `inModelParams`:

  Parameter grid from the model settings

------------------------------------------------------------------------

### Method `getGridSearchPredictions()`

Gets the grid search results from the training cache

#### Usage

    trainingCache$getGridSearchPredictions()

------------------------------------------------------------------------

### Method `isFull()`

Check if cache is full

#### Usage

    trainingCache$isFull()

------------------------------------------------------------------------

### Method `getLastGridSearchIndex()`

Gets the last index from the cached grid search

#### Usage

    trainingCache$getLastGridSearchIndex()

------------------------------------------------------------------------

### Method `dropCache()`

Remove the training cache from the analysis path

#### Usage

    trainingCache$dropCache()

------------------------------------------------------------------------

### Method `trimPerformance()`

Trims the performance of the hyperparameter results by removing the
predictions from all but the best performing hyperparameter

#### Usage

    trainingCache$trimPerformance(hyperparameterResults)

#### Arguments

- `hyperparameterResults`:

  List of hyperparameter results

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    trainingCache$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
