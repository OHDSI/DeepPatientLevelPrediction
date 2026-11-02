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

# Temporary workaround for #191. cblas_sgemm_batch crashes on small matrices in
# DYNAMIC_ARCH builds of OpenBLAS without the fix from OpenBLAS PR #5980, such
# as 0.3.32 in Ubuntu 26.04. torch also exports this symbol, but R has already
# loaded OpenBLAS, so torch calls that copy unless imported with RTLD_DEEPBIND.
# Remove this once supported distributions ship the fix. Import failures are
# ignored here so that they are reported by the delayed torch import instead.
.importTorchWithDeepBind <- function(...) {
  reticulate::py_run_string(
    "
import os, sys
if hasattr(os, 'RTLD_DEEPBIND') and 'torch' not in sys.modules:
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | os.RTLD_DEEPBIND)
    try:
        import torch
    except Exception:
        pass
    finally:
        sys.setdlopenflags(flags)
",
    local = TRUE,
    convert = FALSE
  )
  invisible()
}

# Package hooks run before covr can instrument the namespace. # nocov start
.onLoad <- function(libname, pkgname) {
  torch <<- .initializePythonBindings()
  # The Python modules in inst/python import torch themselves, so this has to
  # run before any of them, not when the delayed torch module is first used.
  if (reticulate::py_available(initialize = FALSE)) {
    .importTorchWithDeepBind()
  } else {
    setHook("reticulate.onPyInit", .importTorchWithDeepBind)
  }
}
# nocov end
