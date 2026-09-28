# DeepPatientLevelPrediction Installation Guide

## Introduction

This vignette describes how to install the Observational Health Data
Sciences and Informatics (OHDSI) DeepPatientLevelPrediction package on
Windows, macOS, and Linux.

## Software Prerequisites

### Windows Users

Under Windows, install:

- R (<https://cran.r-project.org/>) - R \>= 4.1.0; the latest release is
  recommended
- Python - Python 3.14 is recommended; Python \>= 3.10 is supported
- An R development environment, if desired
- Java when using a JDBC-based database connection
- [Rtools](https://cran.r-project.org/bin/windows/Rtools/) when
  installing packages from source

### Mac/Linux Users

Under macOS and Linux, install:

- R (<https://cran.r-project.org/>) - R \>= 4.1.0; the latest release is
  recommended
- Python - Python 3.14 is recommended; Python \>= 3.10 is supported
- An R development environment, if desired
- Java when using a JDBC-based database connection
- Xcode command line tools on macOS when installing packages from source
  (`xcode-select --install`)

## Installing the Package

Install the released package from CRAN:

``` r

install.packages("DeepPatientLevelPrediction")
```

The development version can instead be installed from GitHub.
Development versions may contain changes that have not yet been
released.

``` r

install.packages("remotes")
remotes::install_github("OHDSI/DeepPatientLevelPrediction@develop")
```

### Python environment

Model training and inference use PyTorch through `reticulate`. Python is
not required to install or load the R package. When Python functionality
is first used, `reticulate` can create a managed environment from the
package requirements on systems with internet access.

You can verify the active interpreter with:

``` r

library(DeepPatientLevelPrediction)
reticulate::py_config()
```

Advanced users, users with strict reproducibility requirements, or users
in air-gapped environments can manage the Python environment themselves
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

Accessing the `torch` helper or starting model training triggers
`reticulate` to resolve the Python requirements if `RETICULATE_PYTHON`
is not configured.

``` r

library(DeepPatientLevelPrediction)
torch$randn(10L)
```

This should print out a tensor with ten different values.

On Windows, close other R sessions that are using
`DeepPatientLevelPrediction` or its dependencies before updating the
package; open sessions can lock installed files.

## Testing Installation

``` r

library(DeepPatientLevelPrediction)

torch$randn(10L)
```

The R-side model settings can be created without initializing Python:

``` r

modelSettings <- DeepPatientLevelPrediction::setResNet(
  numLayers = 2L,
  sizeHidden = 64L,
  hiddenFactor = 1L,
  residualDropout = 0,
  hiddenDropout = 0.2,
  sizeEmbedding = 64L,
  estimatorSettings = DeepPatientLevelPrediction::setEstimator(
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
vignette](https://ohdsi.github.com/DeepPatientLevelPrediction/articles/FirstModel.md).

## Acknowledgments

Considerable work has been dedicated to providing the
`DeepPatientLevelPrediction` package.

``` r

citation("DeepPatientLevelPrediction")
```

    ## To cite package 'DeepPatientLevelPrediction' in publications use:
    ## 
    ##   Fridgeirsson E, Reps J, Chan You S, Kim C, John H (2026).
    ##   _DeepPatientLevelPrediction: Deep Learning for Patient-Level
    ##   Prediction_. R package version 2.4.0,
    ##   <https://ohdsi.github.io/DeepPatientLevelPrediction/>.
    ## 
    ## A BibTeX entry for LaTeX users is
    ## 
    ##   @Manual{,
    ##     title = {DeepPatientLevelPrediction: Deep Learning for Patient-Level Prediction},
    ##     author = {Egill Fridgeirsson and Jenna Reps and Seng {Chan You} and Chungsoo Kim and Henrik John},
    ##     year = {2026},
    ##     note = {R package version 2.4.0},
    ##     url = {https://ohdsi.github.io/DeepPatientLevelPrediction/},
    ##   }

**Please reference this paper if you use the PLP Package in your work:**

Reps JM, Schuemie MJ, Suchard MA, Ryan PB, Rijnbeek PR. Design and
implementation of a standardized framework to generate and evaluate
patient-level prediction models using observational healthcare data. J
Am Med Inform Assoc. 2018;25(8):969-975. <doi:10.1093/jamia/ocy032>.
