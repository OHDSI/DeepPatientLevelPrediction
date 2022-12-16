DeepPatientLevelPrediction (develop)
======================
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