DeepPatientLevelPrediction
======================

[![Build Status](https://github.com/OHDSI/DeepPatientLevelPrediction/workflows/R-CMD-check/badge.svg)](https://github.com/OHDSI/DeepPatientLevelPrediction/actions?query=workflow%3AR-CMD-check?branch=main)
[![codecov.io](https://codecov.io/github/OHDSI/DeepPatientLevelPrediction/coverage.svg?branch=main)](https://app.codecov.io/github/OHDSI/DeepPatientLevelPrediction?branch=main)


Introduction
============

DeepPatientLevelPrediction is an R package for building and validating deep learning patient-level predictive models using data in the OMOP Common Data Model format and OHDSI PatientLevelPrediction framework.

Reps JM, Schuemie MJ, Suchard MA, Ryan PB, Rijnbeek PR. Design and implementation of a standardized framework to generate and evaluate patient-level prediction models using observational healthcare data. J Am Med Inform Assoc. 2018;25(8):969-975. doi:10.1093/jamia/ocy032.


Features
========
- Adds deep learning models to use in the OHDSI PatientLevelPrediction framework.
- Allows to add custom deep learning models.
- Includes MLP, ResNet, Transformer, and RealMLP models.
- Allows to use all the features of [PatientLevelPrediction](https://github.com/OHDSI/PatientLevelPrediction/) to validate and explore your model performance.


Technology
==========
DeepPatientLevelPrediction is an R package. It uses Python [PyTorch](https://pytorch.org/) through [reticulate](https://rstudio.github.io/reticulate/) for deep learning model training and inference.

System Requirements
===================
Requires R (version 4.1.0 or higher). Installation from source on Windows may require [Rtools](https://cran.r-project.org/bin/windows/Rtools/). Python 3.10 or newer is required for model training and inference. A CPU can be used for small models; an NVIDIA GPU is recommended for larger deep learning model development.


Getting Started
===============

- To install the package, read the [package installation guide](vignettes/Installing.Rmd).
- Python requirements are declared to `reticulate` when the package loads and
  resolved on first Python use. Advanced users can point `RETICULATE_PYTHON` at
  a prebuilt environment for offline or controlled deployments.
- Please read the main vignette for the package:
[Building Deep Learning Models](vignettes/BuildingDeepModels.Rmd)

User Documentation
==================
Documentation can be found on the [package website](https://ohdsi.github.io/DeepPatientLevelPrediction/).

PDF versions of the documentation are also available, as mentioned above.

Support
=======
* Developer questions/comments/feedback: <a href="https://forums.ohdsi.org/c/developers/7">OHDSI Forum</a>
* We use the <a href="https://github.com/OHDSI/DeepPatientLevelPrediction/issues">GitHub issue tracker</a> for all bugs/issues/enhancements

Contributing
============
Read [here](https://ohdsi.github.io/Hades/contribute.html) how you can contribute to this package.

License
=======
DeepPatientLevelPrediction is licensed under Apache License 2.0

Development
===========
DeepPatientLevelPrediction is being developed in R Studio.
