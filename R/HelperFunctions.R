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

# defuse functions in settings objects so that they are kept as is and then
# evaluated when running the study. Useful when settings are saved as json's
# # to be loaded later before running e.g. Strategus studies.
# defuseCallables <- function(settings) {
#   
#   for (set in names(settings)) {
#     if (is.function(settings[[set]])) {
#       defused <- rlang::enquo(setting[[set]])
#       defusedFun <- function() rlang::eval_tidy(defused)
#       settings[[set]] <- defusedFun
#     } else if (is.list(settings[[set]]) && is.function(settings[[set]]$fun)) {
#       settings[[set]] <- defuse(settings[[set]]$fun)
#     }
#   }
#  return(settings) 
# }
  
#   defusedOptimizer <- rlang::enquo(optimizer)
#   estimatorSettings$optimizer <- function() rlang::eval_tidy(defusedOptimizer)
#   
#   defusedCriterion <- rlang::enquo(criterion)
#   estimatorSettings$criterion <- function() rlang::eval_tidy(defusedCriterion)
#   
#   schedulerFun <- scheduler$fun
#   defusedSchedulerFun <- rlang::enquo(schedulerFun)
#   estimatorSettings$scheduler$fun <- function() rlang::eval_tidy(defusedSchedulerFun)
#   estimatorSettings$scheduler$params<- scheduler$params
#   
#   for (set in names(estimatorSettings)) {
#     if (is.function(estimatorSettings[[set]])) {
#       class(estimatorSettings[[set]]) <- c("delayed", "function")
#     }
#     
#     if (is.list(estimatorSettings[[set]]) && 
#         !is.null(estimatorSettings[[set]]$fun) && 
#         is.function(estimatorSettings[[set]]$fun)) {
#       class(estimatorSettings[[set]]$fun) <- c("delayed", "function")
#     }
#   }  
# }

# defuse and wrap in function one member of a settings object
defuse <- function(setting) {
  
  defusedSetting <- rlang::enquo(setting)
  results <- function() rlang::eval_tidy(defusedSetting)
  class(results) <- c("delayed", class(results))
  
  return(results)
}