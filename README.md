DeepPatientLevelPrediction
======================

[![Build Status](https://github.com/OHDSI/DeepPatientLevelPrediction/workflows/R-CMD-check/badge.svg)](https://github.com/OHDSI/DeepPatientLevelPrediction/actions?query=workflow%3AR-CMD-check?branch=main)
[![codecov.io](https://codecov.io/github/OHDSI/DeepPatientLevelPrediction/coverage.svg?branch=main)](https://codecov.io/github/OHDSI/DeepPatientLevelPrediction?branch=main)


Introduction
============

DeepPatientLevelPrediction is an R package for building and validating deep learning patient-level predictive models using data in the OMOP Common Data Model format and OHDSI PatientLevelPrediction framework.  

Reps JM, Schuemie MJ, Suchard MA, Ryan PB, Rijnbeek PR. [Design and implementation of a standardized framework to generate and evaluate patient-level prediction models using observational healthcare data.](https://academic.oup.com/jamia/article/25/8/969/4989437) J Am Med Inform Assoc. 2018;25(8):969-975.


Features
========
- Adds deep learning models to use in the OHDSI PatientLevelPrediction framework.
- Allows to add custom deep learning models.
- Includes an MLP, ResNet and a Transformer
- Allows to use all the features of [PatientLevelPrediction](https://github.com/OHDSI/PatientLevelPrediction/) to validate and explore your model performance.


Technology
==========
DeepPatientLevelPrediction is an R package. It uses [torch in R](https://torch.mlverse.org/) to build deep learning models without using python.

System Requirements
===================
Requires R (version 4.0.0 or higher). Installation on Windows requires [RTools](http://cran.r-project.org/bin/windows/Rtools/). For training deep learning models in most cases an nvidia GPU is required using either Windows or Linux.


Getting Started
===============

- To install the package please read the [Package installation guide](https://ohdsi.github.io/DeepPatientLevelPrediction/articles/Installing.html)
- Please read the main vignette for the package:
[Building Deep Learning Models](https://ohdsi.github.io/DeepPatientLevelPrediction/articles/BuildingDeepModels.html)

User Documentation
==================
Documentation can be found on the [package website](https://ohdsi.github.io/DeepPatientLevelPrediction).

PDF versions of the documentation are also available, as mentioned above.

Support
=======
* Developer questions/comments/feedback: <a href="http://forums.ohdsi.org/c/developers">OHDSI Forum</a>
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

