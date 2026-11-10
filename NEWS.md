DeepPatientLevelPrediction 2.4.0.9999
======================

New features
- Add a `min_epochs` parameter to early stopping, so that training runs for at least that many epochs before it can stop early (#36).

Bug fixes
- Avoid a segfault in batched matrix multiplication on Ubuntu 26.04. On Linux, torch is now imported with `RTLD_DEEPBIND` so that it does not call the system OpenBLAS, which has a crash in `cblas_sgemm_batch`. This is a temporary workaround until distributions ship the OpenBLAS fix. A startup message explains the risk if torch was already imported before the package loaded (#191).
- Normalize custom embedding concept IDs for Polars 2 (#189).
- Restore the weights of the best epoch after training. Checkpoints shared storage with the live model, so cross-validation predictions came from the last epoch.
- Select the automatic learning rate within each fold on training data only, so that held-out outcomes no longer influence it.
- Invalidate the training cache when the data, outcomes, folds, feature mapping, estimator settings, tuning metric, or the contents of finetuning and custom embedding files change, not only the parameter grid. Caches from earlier versions are recomputed.
- Use the tuning metric and direction requested through `hyperparameterSettings` when selecting hyperparameters and trimming the cache. Its `search`, `sampleSize`, `randomSeed` and `generator` settings are not used, and a warning now says so; the search is set in the model settings.
- Give tied positive and negative predictions half credit in the AUC used for early stopping and epoch selection.
- Clear gradients between learning rate finder trials.
- Train the final model when the data is smaller than one batch, instead of silently returning an untrained model.
- Report training and validation loss per observation when the last batch is incomplete.
- Keep patients without any retained covariates when training and predicting, and raise an error if predictions do not match the cohort.
- Pass the `beta2` setting of RealMLP to the optimizer. AdamW used its default of 0.999 instead of RealMLP's 0.95, so RealMLP results change.
- Assign custom embeddings to the right covariates. Vectors were shifted to the preceding column, and depended on the order of the embeddings file.
- Make predictions independent of the padded sequence width for custom embedding models, RealMLP with mean aggregation and convolutional positional encodings.
- Compute time token gaps using the `timePart` and `timeInterval` of the time reference, and raise an error if these are unavailable.
- Keep column and covariate IDs apart when loading RealMLP wide coefficients.
- Leave frozen RealMLP layers unchanged during data-dependent initialization.
- Use the requested `sigma` in `TemporalPE` when it is not learnable. It was applied as exp(sigma), so the default of 1 acted as 2.72.

Runtime and compatibility
- Require Python 3.11 or newer. Python 3.10 reached end-of-life in October 2026, and duckdb 2.0.0 requires Python 3.11.
- Require PatientLevelPrediction 6.6.0 or newer, which added tuning metrics.
- Models trained with custom embeddings, RealMLP mean aggregation, time tokens, convolutional positional encodings or a fixed `TemporalPE` sigma compute their inputs differently now. Predictions from such saved models change, so retrain them.

Documentation and CI
- Add an Ubuntu 26.04 CI job and a regression test for batched matrix multiplication.
- Stop pinning Python duckdb on macOS CI, because version 1.5.2 cannot read files written by R duckdb 2.0.0.
- Update `tlmgr` before installing LaTeX packages in the R-devel CI job.

DeepPatientLevelPrediction 2.4.0
======================

New features
- Add hyperparameter tuning support for RealMLP.

Bug fixes
- Keep temporal dataset `feature_ids`, `feature_values`, and `time_ids` aligned when multiple features share the same time ID.

Documentation and CI
- Refresh RealMLP examples for current grid/random tuning behavior.
- Add a dedicated temporal transformer vignette.
- Update CI and installation guidance to validate the recommended Python environment with Python 3.14 and torch 2.12.1.
- Pin Python duckdb to 1.5.2 on macOS CI to avoid intermittent shutdown segfaults.
- Limit push-triggered R CMD checks to `develop` and `main` while keeping pull request checks enabled for all branches.
- Refresh vignette wording and examples for current runtime guidance.

DeepPatientLevelPrediction 2.3.0
======================

New features
- Add core `RealMLP` support for tabular deep learning, including paper-style numeric embeddings, schedules, and data-dependent initialization (PR #171)
- Add dropout support to `TemporalPE` and test coverage for positional-encoding dropout behavior (PR #165)

Bug fixes
- Replace deprecated `pynvml` with `nvidia-ml-py` (PR #167, closes #166)
- Improve expand-component handling and add regression tests (PR #164)
- Fix estimator seed casting, prediction shaping, and custom-embedding / Poincare model paths for current torch and reticulate behavior (PR #171)

Runtime and compatibility
- Relax the torch requirement to `torch>=2.7,<3` and validate against `torch==2.10.0` in stable CI lanes (PR #171)
- Add a Python 3.10 compatibility lane using the latest available torch (PR #171)
- Clean `torchinductor_runner` artifacts during `R CMD check` (PR #171)

Documentation and CI
- Update installation guidance to recommend `uv` and Python 3.12 (PR #171)
- Update GitHub Actions checks to use Python 3.12 by default and fix Windows torch installation handling (PR #171)
- Update pkgdown link checking for current `lychee` CLI behavior (PR #171)

DeepPatientLevelPrediction 2.2.0
======================
- [Feature] Add positional encodings that use `timeId` (PR #154):
    - `SinusoidalPE`: Absolute sinusoidal positional embeddings using
    - `LearnablePE`: Learnable absolute positional embeddings 
    - `TapePE`: Time-absolute Positional Encodings (`tAPE`)
    - `RotaryPE`: Rotary position embeddings (`RoPE`)
    - `RelativePE`: Relative positional embeddings (`RPE`)
    - `EfficientRPE`: Efficient relative positional embeddings (`eRPE`)
    - `TemporalPE`: Combines an absolute temporal embedding and a semantic one
    - `StochasticConvPE`: Relative learnable positional embedding (`convSPE`)
    - `HybridRoPEconvPE`: Combines `RoPE` and `convSPE`
    - `TUPE`: Transformer with learnable positional encoding (`TUPE`)
    - `ALiBiPE`: Attention with linear biases (`ALiBi`)
- [Feature] Use `py_require` from reticulate to manage python dependencies and update min requirements (PR #150)
- [Internal] Refactor transformer/dataset/embedding classes to use the same code whether temporal or not (PR #147)
- [Feature] Use train/validation split for model selection instead of cross validation (PR #145)
- [Feature] Temporal transformer added which supports RoPE and time tokens (PR #147)
- [Feature] Temporal data processing added (PR #147)
- [CI] Use uv for python in github actions (PR #136)
- [Feature] Add an option to use torch compile (PR #133)
- [Feature] More efficient conversions from polars to torch in dataset processing (PR #133)
- [CI] Automatically detect broken links in docs using github actions
- [Feature] Initialize models with embeddings, including Poincare (PR #125 and PR #130)
- [Feature] Model initialization made more flexible with classes (PR #124)

DeepPatientLevelPrediction 2.1.0
======================
  - Added basic transfer learning functionality. See vignette("TransferLearning")
  - Add a gpu memory cleaner to clean cached memory after out of memory error
  - The python module torch is now accessed through an exported function instead of loading the module at package load
  - Added gradient accumulation. Studies running at different sites using different hardware can now use same effective batch size by accumulating gradients.
  - Refactored out the cross validation from the hyperparameter tuning
  - Remove predictions from non-optimal hyperparameter combinations to save space
  - Only use html vignettes 
  - Rename MLP to MultiLayerPerceptron
  

DeepPatientLevelPrediction 2.0.3
======================
  - Hotfix: Fix count for polars v0.20.x
  
DeepPatientLevelPrediction 2.0.2
======================
  - Ensure output from predict_proba is numeric instead of 1d array
  - Refactoring: Move cross-validation to a separate function
  - Refactoring: Move paramsToTune to a separate function 
  - linting: Enforcing HADES style
  - Calculate AUC ourselves with torch, get rid of scikit-learn dependency
  - added Andromeda to dev dependencies


DeepPatientLevelPrediction 2.0.1
======================
  - Connection parameter fixed to be in line with newest polars
  - Fixed a bug where LRFinder used a hardcoded batch size
  - Seed is now used in LRFinder so it's reproducible
  - Fixed a bug in NumericalEmbedding
  - Fixed a bug for Transformer and numerical features
  - Fixed a bug when resuming from a full TrainingCache (thanks Zoey Jiang and Linying Zhang )
  - Updated installation documentation after feedback from HADES hackathon
  - Fixed a bug where order of numeric features wasn't conserved between training and test set
  - TrainingCache now only saves prediction dataframe for the best performing model 

DeepPatientLevelPrediction 2.0.0
======================
  - New backend which uses PyTorch through reticulate instead of torch in R
  - All models ported over to python
  - Dataset class now in python
  - Estimator class in python
  - Learning rate finder in python
  - Added input checks and tests for wrong inputs
  - Training-cache for single hyperparameter combination added
  - Fixed empty test for training-cache

DeepPatientLevelPrediction 1.1.6
======================
  - Caching and resuming of hyperparameter iterations

DeepPatientLevelPrediction 1.1.5
======================
  - Fix bug where device function was not working for LRFinder

DeepPatientLevelPrediction 1.1.4
======================
 - Remove torchopt dependency since AdamW is now in torch
 - Update torch dependency to >=0.10.0
 - Allow device to be a function that resolves during Estimator initialization

DeepPatientLevelPrediction 1.1.3
======================
- Fix actions after torch updated to v0.10 (#65)

DeepPatientLevelPrediction 1.1.2
======================
- Fix bug introduced by removing modelType from attributes (#59)

DeepPatientLevelPrediction 1.1
======================
- Check for if number of heads is compatible with embedding dimension fixed (#55)
- Now transformer width can be specified as a ratio of the embedding dimensions (dimToken), (#53)
- A custom metric can now be defined for earlyStopping and learning rate schedule (#51)
- Added a setEstimator function to configure the estimator (#51)
- Seed added for model weight initialization to improve reproducibility (#51)
- Added a learning rate finder for automatic calculation of learning rate (#51)
- Add seed for sampling hyperparameters (#50)
- used vectorised torch operations to speed up data conversion in torch dataset

DeepPatientLevelPrediction 1.0.2
======================
- Fix torch binaries issue when running tests from other github actions
- Fix link on website
- Fix tidyselect to silence warnings.

DeepPatientLevelPrediction 1.0.1
======================
- Added changelog to website
- Added a first model tutorial
- Fixed small bug in default ResNet and Transformer

DeepPatientLevelPrediction 1.0.0
======================
- created an Estimator R6 class to handle the model fitting
- Added three non-temporal models. An MLP, a ResNet and a Transformer
- ResNet and Transformer have default versions of hyperparameters
- Created tests and documentation for the package
