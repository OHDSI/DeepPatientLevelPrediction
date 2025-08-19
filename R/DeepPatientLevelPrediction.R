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

#' Pytorch module
#'
#' The `torch` module object is the equivalent of
#' `reticulate::import("torch")` and provided mainly as a convenience.
#'
#' @returns the torch Python module
#' @export
#' @usage NULL
#' @format An object of class `python.builtin.module`
torch <- NULL

.onLoad <- function(libname, pkgname) {
  reticulate::py_require(c(
    "polars>=1.31.0", "pyarrow", "duckdb", "torch<=2.6.0", "tqdm",
    "pynvml"
  ))
  torch <<- reticulate::import("torch", delay_load = TRUE)
}
