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
  - [Internal] Refactor transformer/dataset/embedding classes to use same code wether temporal or not (PR #147)
  - [Feature] Use train/validation split for model selection instead of cross validation (PR #145)
  - [Feature] Temporal transformer added which supports RopE and time tokens (PR #147)
  - [Feature] Temporal data processing added (PR #147)
  - [CI] Use uv for python in github actions (PR #136)
  - [Feature] Add an option to use torch compile (PR #133)
  - [Feature] More efficient conversions from polars to torch in dataset processing (PR #133)
  - [CI] Automatically detect broken links in docs using github actions
  - [Feature] Model initialization made more flexible with classes

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
  - Calculate AUC ourselves with torch, get rid of scikit-learn dependancy
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
  - New backend which uses pytorch through reticulate instead of torch in R
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
 - Remove torchopt dependancy since adamw is now in torch
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
- Added a learning rate finder for automatic calculatio of learning rate (#51)
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
