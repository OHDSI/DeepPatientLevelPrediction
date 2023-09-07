# @file HelperFunctions.R
#
# Copyright 2023 Observational Health Data Sciences and Informatics
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
#' Convert a camel case string to snake case
#'
#' @param string   The string to be converted
#'
#' @return
#' A string
#'
camelCaseToSnakeCase <- function(string) {
  string <- gsub("([A-Z])", "_\\1", string)
  string <- tolower(string)
  string <- gsub("([a-z])([0-9])", "\\1_\\2", string)
  return(string)
}

#' Convert the names of an object from camel case to snake case
#'
#' @param object   The object of which the names should be converted
#'
#' @return
#' The same object, but with converted names.
camelCaseToSnakeCaseNames <- function(object) {
  names(object) <- camelCaseToSnakeCase(names(object))
  return(object)
}