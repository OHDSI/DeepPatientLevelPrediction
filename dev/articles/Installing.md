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
- Python - Recommend Python 3.14. Python \>= 3.10 is supported
- RStudio (<https://www.rstudio.com/> )
- Java (<http://www.java.com> )
- RTools (<https://cran.r-project.org/bin/windows/Rtools/>)

### Mac/Linux Users

Under Mac and Linux the OHDSI DeepPLP package requires installing:

- R (<https://cran.r-project.org/> ) - (R \>= 4.0.0, but latest is
  recommended)
- Python - Recommend Python 3.14. Python \>= 3.10 is supported
- RStudio (<https://www.rstudio.com/> )
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

### Python environment

Since the package uses PyTorch through `reticulate`, a working Python
environment is required. For most users on a computer with internet
access, no manual Python setup is needed: loading or using
`DeepPatientLevelPrediction` should let `reticulate` create a managed
environment from the package requirements.

You can verify the active interpreter with:

``` r

library(DeepPatientLevelPrediction)
reticulate::py_config()
```

Advanced users, users with strict reproducibility requirements, or users
in airgapped environments can manage the Python environment themselves
and tell `reticulate` which interpreter to use. One option is to create
the environment with `uv` and Python 3.14:

``` bash
uv python install 3.14
uv venv --python 3.14
uv pip install polars tqdm pyarrow duckdb nvidia-ml-py numpy
uv pip install "torch==2.12.1" --index https://download.pytorch.org/whl/cpu/
```

The second `uv pip install` command installs the CPU build of PyTorch.
If you want to train on a GPU, install the PyTorch build that matches
your CUDA setup instead.

To force `reticulate` to use a manually managed interpreter, set
`RETICULATE_PYTHON` in `.Renviron`.

For Linux/macOS:

    RETICULATE_PYTHON="/path/to/project/.venv/bin/python"

For Windows:

    RETICULATE_PYTHON="C:/path/to/project/.venv/Scripts/python.exe"

Then restart your R session.

Python 3.9 is end-of-life and should not be used. Python 3.10 is still
supported, but Python 3.14 is recommended.

### Installing DeepPatientLevelPrediction using remotes

To install using `remotes` run:

``` r

install.packages("remotes")
remotes::install_github("OHDSI/DeepPatientLevelPrediction")
```

Loading the package or using the `torch` helper should trigger
`reticulate` to resolve the Python requirements if you have not
configured `RETICULATE_PYTHON`.

``` r

library(DeepPatientLevelPrediction)
torch$randn(10L)
```

This should print out a tensor with ten different values.

When installing make sure to close any other RStudio sessions that are
using `DeepPatientLevelPrediction` or any dependency. Keeping RStudio
sessions open can cause locks on Windows that prevent the package
installing.

## Testing Installation

``` r

library(DeepPatientLevelPrediction)

torch$randn(10L)

modelSettings <- setResNet(
  numLayers = 2L,
  sizeHidden = 64L,
  hiddenFactor = 1L,
  residualDropout = 0,
  hiddenDropout = 0.2,
  sizeEmbedding = 64L,
  estimatorSettings = setEstimator(
    learningRate = 3e-4,
    weightDecay = 1e-6,
    device = "cpu",
    batchSize = 128L,
    epochs = 3L,
    seed = 42L
  ),
  hyperParamSearch = "random",
  randomSample = 1L
)

stopifnot(inherits(modelSettings, "modelSettings"))
```

To run an end-to-end patient-level prediction example, continue with the
[first-model
vignette](https://ohdsi.github.com/DeepPatientLevelPrediction/dev/articles/FirstModel.md).

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
    ##   version 2.3.0.9999,
    ##   <https://github.com/OHDSI/DeepPatientLevelPrediction>.
    ## 
    ## A BibTeX entry for LaTeX users is
    ## 
    ##   @Manual{,
    ##     title = {DeepPatientLevelPrediction: Deep Learning for Patient Level Prediction Using Data in the
    ## OMOP Common Data Model},
    ##     author = {Egill Fridgeirsson and Jenna Reps and Seng {Chan You} and Chungsoo Kim and Henrik John},
    ##     year = {2026},
    ##     note = {R package version 2.3.0.9999},
    ##     url = {https://github.com/OHDSI/DeepPatientLevelPrediction},
    ##   }

**Please reference this paper if you use the PLP Package in your work:**

[Reps JM, Schuemie MJ, Suchard MA, Ryan PB, Rijnbeek PR. Design and
implementation of a standardized framework to generate and evaluate
patient-level prediction models using observational healthcare data. J
Am Med Inform Assoc.
2018;25(8):969-975.](http://dx.doi.org/10.1093/jamia/ocy032)
