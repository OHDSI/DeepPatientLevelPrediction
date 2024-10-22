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

#' Convert a camel case string to snake case
#'
#' @param string   The string to be converted
#'
#' @return
#' A string
#'
snakeCaseToCamelCase <- function(string) {
  string <- tolower(string)
  for (letter in letters) {
    string <- gsub(paste("_", letter, sep = ""), toupper(letter), string)
  }
  string <- gsub("_([0-9])", "\\1", string)
  return(string)
}

#' Convert the names of an object from snake case to camel case
#'
#' @param object   The object of which the names should be converted
#'
#' @return
#' The same object, but with converted names.
snakeCaseToCamelCaseNames <- function(object) {
  names(object) <- snakeCaseToCamelCase(names(object))
  return(object)
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

#' helper function to check class of input
#'
#' @param parameter the input parameter to check
#' @param classes which classes it should belong to (one or more)
checkIsClass <- function(parameter, classes) {
  name <- deparse(substitute(parameter))
  if (!inherits(x = parameter, what = classes)) {
    ParallelLogger::logError(paste0(name, " should be of class:", classes, " "))
    stop(paste0(name, " is wrong class"))
  }
  return(TRUE)
}

#' helper function to check that input is higher than a certain value
#'
#' @param parameter the input parameter to check, can be a vector
#' @param value which value it should be higher than
checkHigher <- function(parameter, value) {
  name <- deparse(substitute(parameter))
  if (!is.numeric(parameter) || all(parameter == value)) {
    ParallelLogger::logError(paste0(name, " needs to be > ", value))
    stop(paste0(name, " needs to be > ", value))
  }
  return(TRUE)
}

#' helper function to check that input is higher or equal than a certain value
#'
#' @param parameter the input parameter to check, can be a vector
#' @param value which value it should be higher or equal than
checkHigherEqual <- function(parameter, value) {
  name <- deparse(substitute(parameter))
  if (!is.numeric(parameter) || all(parameter < value)) {
    ParallelLogger::logError(paste0(name, " needs to be >= ", value))
    stop(paste0(name, " needs to be >= ", value))
  }
  return(TRUE)
}

#' helper function to check if a file exists
#' @param file the file to check
checkFileExists <- function(file) {
  if (!file.exists(file)) {
    ParallelLogger::logError(paste0("File ", file, " does not exist"))
    stop(paste0("File ", file, " does not exist"))
  }
  return(TRUE)
}
