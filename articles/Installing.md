# DeepPatientLevelPrediction Installation Guide

## Introduction

This vignette describes how you need to install the Observational Health
Data Science and Informatics (OHDSI) DeepPatientLevelPrediction under
Windows, Mac and Linux.

## Software Prerequisites

### Windows Users

Under Windows the OHDSI Deep Patient Level Prediction (DeepPLP) package
requires installing:

- R (<https://cran.r-project.org/> ) - (R \>= 4.0.0, but latest is
  recommended)
- Python - Recommend Python 3.12. Python \>= 3.10 is supported
- Rstudio (<https://www.rstudio.com/> )
- Java (<http://www.java.com> )
- RTools (<https://cran.r-project.org/bin/windows/Rtools/>)

### Mac/Linux Users

Under Mac and Linux the OHDSI DeepPLP package requires installing:

- R (<https://cran.r-project.org/> ) - (R \>= 4.0.0, but latest is
  recommended)
- Python - Recommend Python 3.12. Python \>= 3.10 is supported
- Rstudio (<https://www.rstudio.com/> )
- Java (<http://www.java.com> )
- Xcode command line tools(run in terminal: xcode-select –install) \[MAC
  USERS ONLY\]

## Installing the Package

The preferred way to install the package is by using `remotes`, which
will automatically install the latest release and all the latest
dependencies.

If you do not want the official release you could install the bleeding
edge version of the package (latest develop branch).

Note that the latest develop branch could contain bugs, please report
them to us if you experience problems.

### Installing Python environment

Since the package uses `pytorch` through `reticulate`, a working Python
installation is required. We recommend using `uv` with Python 3.12.

Install `uv` by following the official instructions:

<https://docs.astral.sh/uv/getting-started/installation/>

Then create a local virtual environment for this project:

``` bash
uv python install 3.12
uv venv --python 3.12
```

Tell `reticulate` to use that interpreter by setting `RETICULATE_PYTHON`
in `.Renviron`.

For Linux/macOS:

    RETICULATE_PYTHON="/path/to/project/.venv/bin/python"

For Windows:

    RETICULATE_PYTHON="C:/path/to/project/.venv/Scripts/python.exe"

Then restart your R session. You can verify the active interpreter with:

``` r
reticulate::py_config()
```

Python 3.9 is end-of-life and should not be used. Python 3.10 is still
supported, but Python 3.12 is recommended.

### Installing DeepPatientLevelPrediction using remotes

To install using `remotes` run:

``` r
install.packages("remotes")
remotes::install_github("OHDSI/DeepPatientLevelPrediction")
```

This should install the required python packages. If that doesn’t happen
it can be triggered by calling:

    library(DeepPatientLevelPrediction)
    torch$randn(10L)

This should print out a tensor with ten different values.

When installing make sure to close any other Rstudio sessions that are
using `DeepPatientLevelPrediction` or any dependency. Keeping Rstudio
sessions open can cause locks on windows that prevent the package
installing.

## Testing Installation

``` r
library(PatientLevelPrediction)
library(DeepPatientLevelPrediction)

data(plpDataSimulationProfile)
sampleSize <- 1e3
plpData <- simulatePlpData(
  plpDataSimulationProfile,
  n = sampleSize 
)

populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
                                                          requireTimeAtRisk = F, 
                                                          riskWindowStart = 1, 
                                                          riskWindowEnd = 365)
# a very simple resnet
modelSettings <- setResNet(numLayers = 2L, 
                           sizeHidden = 64L, 
                           hiddenFactor = 1L,
                           residualDropout = 0, 
                           hiddenDropout = 0.2, 
                           sizeEmbedding = 64L, 
                           estimatorSettings = setEstimator(learningRate = 3e-4,
                                                            weightDecay = 1e-6,
                                                            device='cpu',
                                                            batchSize=128L,
                                                            epochs=3L,
                                                            seed = 42),
                           hyperParamSearch = 'random',
                           randomSample = 1L)

plpResults <- PatientLevelPrediction::runPlp(plpData = plpData,
               outcomeId = 3,
               modelSettings = modelSettings,
               analysisId = 'Test',
               analysisName = 'Testing DeepPlp',
               populationSettings = populationSettings,
               splitSettings = createDefaultSplitSetting(),
               sampleSettings = createSampleSettings(), 
               featureEngineeringSettings = createFeatureEngineeringSettings(), 
               preprocessSettings = createPreprocessSettings(),
               logSettings = createLogSettings(),
               executeSettings = createExecuteSettings(runSplitData = TRUE,
                                                      runSampleData = FALSE,
                                                      runFeatureEngineering = FALSE,
                                                      runPreprocessData = TRUE,
                                                      runModelDevelopment = TRUE,
                                                      runCovariateSummary = TRUE
                                                      ))
```

## Acknowledgments

Considerable work has been dedicated to provide the
`DeepPatientLevelPrediction` package.

``` r
citation("DeepPatientLevelPrediction")
```

    ## To cite package 'DeepPatientLevelPrediction' in publications use:
    ## 
    ##   Fridgeirsson E, Reps J, Chan You S, Kim C, John H (2026).
    ##   _DeepPatientLevelPrediction: Deep Learning for Patient Level
    ##   Prediction Using Data in the OMOP Common Data Model_. R package
    ##   version 2.3.0, <https://github.com/OHDSI/DeepPatientLevelPrediction>.
    ## 
    ## A BibTeX entry for LaTeX users is
    ## 
    ##   @Manual{,
    ##     title = {DeepPatientLevelPrediction: Deep Learning for Patient Level Prediction Using Data in the
    ## OMOP Common Data Model},
    ##     author = {Egill Fridgeirsson and Jenna Reps and Seng {Chan You} and Chungsoo Kim and Henrik John},
    ##     year = {2026},
    ##     note = {R package version 2.3.0},
    ##     url = {https://github.com/OHDSI/DeepPatientLevelPrediction},
    ##   }

**Please reference this paper if you use the PLP Package in your work:**

[Reps JM, Schuemie MJ, Suchard MA, Ryan PB, Rijnbeek PR. Design and
implementation of a standardized framework to generate and evaluate
patient-level prediction models using observational healthcare data. J
Am Med Inform Assoc.
2018;25(8):969-975.](http://dx.doi.org/10.1093/jamia/ocy032)
