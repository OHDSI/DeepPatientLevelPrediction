## Release summary

This is the first CRAN submission of DeepPatientLevelPrediction. The package
provides deep learning models for the PatientLevelPrediction framework using
PyTorch through reticulate.

## Test environments

* Ubuntu 24.04.4 LTS, R 4.6.0
* Ubuntu 24.04.4 LTS, Python 3.14.3 and PyTorch 2.12.1
* Ubuntu 24.04.4 LTS, Python 3.14.3 and PyTorch 2.13.0

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

Python is an optional runtime requirement. It is not initialized during package
installation, loading, runnable examples, vignettes, or the CRAN-safe test
suite. The complete Python integration suite, containing 554 test expectations,
is run separately because PyTorch is not assumed to be available on CRAN check
machines.

The package also passes a check with suggested packages unavailable.

## Reverse dependencies

There are no reverse dependencies because this is a new submission.
