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
