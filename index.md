# DeepPatientLevelPrediction

[![Build
Status](https://github.com/OHDSI/DeepPatientLevelPrediction/workflows/R-CMD-check/badge.svg)](https://github.com/OHDSI/DeepPatientLevelPrediction/actions?query=workflow%3AR-CMD-check?branch=main)
[![codecov.io](https://codecov.io/github/OHDSI/DeepPatientLevelPrediction/coverage.svg?branch=main)](https://codecov.io/github/OHDSI/DeepPatientLevelPrediction?branch=main)

# Introduction

DeepPatientLevelPrediction is an R package for building and validating
deep learning patient-level predictive models using data in the OMOP
Common Data Model format and OHDSI PatientLevelPrediction framework.

Reps JM, Schuemie MJ, Suchard MA, Ryan PB, Rijnbeek PR. [Design and
implementation of a standardized framework to generate and evaluate
patient-level prediction models using observational healthcare
data.](https://academic.oup.com/jamia/article/25/8/969/4989437) J Am Med
Inform Assoc. 2018;25(8):969-975.

# Features

- Adds deep learning models to use in the OHDSI PatientLevelPrediction
  framework.
- Allows to add custom deep learning models.
- Includes MLP, ResNet, Transformer, and RealMLP models.
- Allows to use all the features of
  [PatientLevelPrediction](https://github.com/OHDSI/PatientLevelPrediction/)
  to validate and explore your model performance.

# Technology

DeepPatientLevelPrediction is an R package. It uses Python
[PyTorch](https://pytorch.org/) through
[reticulate](https://rstudio.github.io/reticulate/) for deep learning
model training and inference.

# System Requirements

Requires R (version 4.0.0 or higher). Installation on Windows requires
[RTools](http://cran.r-project.org/bin/windows/Rtools/). A CPU can be
used for small examples and tests; an NVIDIA GPU is recommended for
larger deep learning model development.

# Getting Started

- To install the package please read the [Package installation
  guide](https://ohdsi.github.io/DeepPatientLevelPrediction/articles/Installing.html)
- Python dependencies are managed through `reticulate` when the package
  is loaded or used. Advanced users can point `RETICULATE_PYTHON` at a
  prebuilt environment for offline or controlled deployments.
- Please read the main vignette for the package: [Building Deep Learning
  Models](https://ohdsi.github.io/DeepPatientLevelPrediction/articles/BuildingDeepModels.html)

# User Documentation

Documentation can be found on the [package
website](https://ohdsi.github.io/DeepPatientLevelPrediction).

PDF versions of the documentation are also available, as mentioned
above.

# Support

- Developer questions/comments/feedback: [OHDSI
  Forum](http://forums.ohdsi.org/c/developers)
- We use the [GitHub issue
  tracker](https://github.com/OHDSI/DeepPatientLevelPrediction/issues)
  for all bugs/issues/enhancements

# Contributing

Read [here](https://ohdsi.github.io/Hades/contribute.html) how you can
contribute to this package.

# License

DeepPatientLevelPrediction is licensed under Apache License 2.0

# Development

DeepPatientLevelPrediction is being developed in R Studio.
