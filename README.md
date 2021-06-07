DeepPatientLevelPrediction
======================

[![Build Status](https://github.com/OHDSI/DeepPatientLevelPrediction/workflows/R-CMD-check/badge.svg)](https://github.com/OHDSI/DeepPatientLevelPrediction/actions?query=workflow%3AR-CMD-check)
[![codecov.io](https://codecov.io/github/OHDSI/DeepPatientLevelPrediction/coverage.svg?branch=master)](https://codecov.io/github/OHDSI/DeepPatientLevelPrediction?branch=master)


Introduction
============

DeepPatientLevelPrediction is an R package for building and validating deep learning patient-level predictive models using data in the OMOP Common Data Model format and OHDSI PatientLevelPrediction framework.  

Reps JM, Schuemie MJ, Suchard MA, Ryan PB, Rijnbeek PR. [Design and implementation of a standardized framework to generate and evaluate patient-level prediction models using observational healthcare data.](https://academic.oup.com/jamia/article/25/8/969/4989437) J Am Med Inform Assoc. 2018;25(8):969-975.


Features
========
- add



Technology
==========
DeepPatientLevelPrediction is an R package, with some functions implemented in C++ and python.

System Requirements
===================
Requires R (version 3.3.0 or higher). Installation on Windows requires [RTools](http://cran.r-project.org/bin/windows/Rtools/). Libraries used in DeepPatientLevelPrediction require Java and Python.

The python installation is required for some of the machine learning algorithms. We advise to
install Python 3.7 using Anaconda (https://www.continuum.io/downloads). 

Getting Started
===============

- add

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

