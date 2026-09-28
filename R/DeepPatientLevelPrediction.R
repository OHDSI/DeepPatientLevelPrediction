# @file DeepPatientLevelPrediction.R
#
# Copyright 2022 Observational Health Data Sciences and Informatics
#
# This file is part of DeepPatientLevelPrediction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#' DeepPatientLevelPrediction
#'
#' @description A package containing deep learning extensions for developing
#' prediction models using data in the OMOP CDM
#'
#' @name DeepPatientLevelPrediction
#' @importFrom dplyr %>%
#' @importFrom reticulate r_to_py py_to_r
#' @importFrom rlang .data
"_PACKAGE"

# package level global state
.globals <- new.env(parent = emptyenv())

#' PyTorch Module
#'
#' The `torch` module object is the equivalent of
#' `reticulate::import("torch")` and is provided mainly as a convenience.
#' Accessing the module initializes Python and requires the Python dependencies
#' listed in `SystemRequirements` in the package `DESCRIPTION` file.
#'
#' @returns The `torch` Python module.
#' @export
#' @usage NULL
#' @format An object of class `python.builtin.module`
#' @examples
#' \dontrun{
#' # Requires the Python dependencies described in vignette("Installing").
#' torch$randn(10L)
#' }
torch <- NULL

.pythonRequirements <- c(
  "polars>=1.31.0",
  "pyarrow",
  "duckdb",
  "numpy",
  "torch>=2.7,<3",
  "tqdm",
  "nvidia-ml-py"
)

.pythonImportError <- function(error) {
  stop(
    paste(
      "DeepPatientLevelPrediction could not initialize its Python environment.",
      "Install Python 3.10 or newer and the package requirements described in",
      "vignette('Installing', package = 'DeepPatientLevelPrediction').",
      "Original error:",
      conditionMessage(error)
    ),
    call. = FALSE
  )
}

.initializePythonBindings <- function(
    pyRequire = reticulate::py_require,
    pyImport = reticulate::import) {
  pyRequire(
    .pythonRequirements,
    python_version = ">=3.10"
  )
  pyImport(
    "torch",
    delay_load = list(on_error = .pythonImportError)
  )
}

# Package hooks run before covr can instrument the namespace. # nocov start
.onLoad <- function(libname, pkgname) {
  torch <<- .initializePythonBindings()
}
# nocov end
