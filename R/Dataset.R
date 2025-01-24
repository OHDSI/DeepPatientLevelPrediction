# @file Dataset.R
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
createDataset <- function(data, labels, plpModel = NULL) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  dataset <- reticulate::import_from_path("Dataset", path = path)$Data
  if (is.null(attributes(data)$path)) {
    # sqlite object
    attributes(data)$path <- attributes(data)$dbname
  }
  if (is.null(plpModel) && is.null(data$numericalIndex)) {
    data <- dataset(r_to_py(normalizePath(attributes(data)$path)),
                    r_to_py(labels$outcomeCount))
  } else if (!is.null(data$numericalIndex)) {
    numericalIndex <-
      r_to_py(as.array(data$numericalIndex %>% dplyr::pull()))
    data <- dataset(r_to_py(normalizePath(attributes(data)$path)),
                    r_to_py(labels$outcomeCount),
                    numericalIndex)
  } else {
    numericalFeatures <-
      r_to_py(as.array(which(plpModel$covariateImportance$isNumeric)))
    data <- dataset(r_to_py(normalizePath(attributes(data)$path)),
      numerical_features = numericalFeatures
    )
  }

  return(data)
}


#'  A torch dataset for temporal data
#' @export
TemporalDataset <- torch::dataset(
  name = "TemporalDataset",
  #' @param data           a dataframe like object with the covariates
  #' @param labels         a dataframe with the labels
  #' @param datasetInfo    info needed to recreate same dataset on test set
  initialize = function(data, labels = NULL, datasetInfo = NULL) { 
    start <- Sys.time()
    self$datasetInfo <- datasetInfo
    staticData <- data %>% dplyr::filter(is.na(timeId))
    temporalData <- data %>% dplyr::filter(!is.na(timeId))
    
    self$processTarget(labels, data)
    
    self$processStatic(staticData)
    
    self$processTemporal(temporalData)
    
    delta <- Sys.time() - start
    ParallelLogger::logInfo("Data conversion for dataset took ", signif(delta, 3), " ", attr(delta, "units"))
  },
  processTemporal = function(data) {
    if (is.null(self$datasetInfo$temporalFeatures)) {
      numMap <- data %>%
        dplyr::group_by(columnId) %>%
        dplyr::summarise(n = dplyr::n_distinct(.data$covariateValue)) %>%
        dplyr::collect()
      numericalIndex <- numMap %>% dplyr::pull(n) > 1
      
      # add features
      catColumns <- numMap[!numericalIndex, ]$columnId
      
      temporalFeatures <- data %>% 
        dplyr::filter(columnId %in% catColumns) %>% 
        dplyr::select(covariateId) %>% 
        dplyr::distinct() %>% 
        dplyr::pull()
      self$datasetInfo$temporalFeatures <- temporalFeatures
    } else {
      temporalFeatures <- self$datasetInfo$temporalFeatures
    }
    map <- data.frame(covariateId = temporalFeatures,
                      newColumnId = 1:length(temporalFeatures))
    dataCat <- data %>% 
      dplyr::inner_join(map, by = "covariateId") %>%
      dplyr::select(c("rowId", "newColumnId", "timeId")) %>%
      dplyr::rename(columnId = newColumnId) %>%
      dplyr::arrange(rowId, timeId, columnId) %>% 
      dplyr::collect()
    dataCat$timeId <- dataCat$timeId + 2 # 1 will be my padding idx
    catTensor <- torch::torch_tensor(cbind(dataCat$rowId, 
                                           dataCat$timeId, 
                                           dataCat$columnId),
                                     dtype = torch::torch_long())
    
    visitsList <- dataCat %>% 
      dplyr::group_by(rowId) %>% 
      dplyr::select(rowId, timeId) %>% 
      dplyr::distinct(timeId) %>% 
      dplyr::group_split(.keep = FALSE)
    visitsCount <- unlist(lapply(visitsList, function(x) nrow(x)))
    if (is.null(self$datasetInfo$maxVisit)) {
      # TODO add maxVisit as a variable, either between 0-1 or absolute number
      self$maxVisit <- floor(quantile(visitsCount, 0.99)[[1]])
      self$datasetInfo$maxVisit <- self$maxVisit
    } else {
      self$maxVisit <- self$datasetInfo$maxVisit
    }
    self$visits <- lapply(visitsList, function(x) {
      torch::nnf_pad(torch::torch_tensor(x$timeId, dtype = torch::torch_long()),
                     c(0, self$maxVisit - nrow(x)), value = 1L)})
    start <- Sys.time()
    seqs <- list()
    lengths <- list()
    ParallelLogger::logInfo(paste0("Preparing temporal features"))
    progressBar <- utils::txtProgressBar(style = 3)
    for (i in 1:length(self$target)) {
      # get sequence for that rowId
      rowData <- catTensor[catTensor[, 1] == i, ]
      
      uniqueVisits <- torch::torch_unique_consecutive(rowData[, 2])[[1]]
      uniqueVisitsCount <- uniqueVisits$shape[[1]]
      if (uniqueVisitsCount > self$maxVisit) {
        # remove visits that are to old
        maxVisitTimeId <- uniqueVisits[(self$maxVisit)]
        rowData <- rowData[rowData[, 2] <= maxVisitTimeId, ]
      }
      # remap to unique visits instead of timeId
      timeIds <- rowData[, 2]
      uniqueVisitsId <- torch::torch_tensor(1:uniqueVisitsCount, 
        dtype = torch::torch_long())
      remapIndex <- torch::torch_bucketize(timeIds, uniqueVisits) + 1L
      rowData[, 2] <- uniqueVisitsId[remapIndex]
      
      indices <- rowData[, 2:3]$t_()
      values <- torch::torch_ones(indices$shape[[2]], dtype = torch::torch_float32())
      rowMatrix <- torch::torch_sparse_coo_tensor(indices = indices,
                                                  values = values,
                                                  size = c(self$maxVisit, 
                                                           length(temporalFeatures))
                                                  )
      seqs <- append(seqs, rowMatrix)
      lengths <- append(lengths, rowData$shape[[1]])
      utils::setTxtProgressBar(progressBar, i / self$target$shape)
    }
    self$sequences <- seqs
    self$lengths <- torch::torch_tensor(unlist(lengths), dtype = torch::torch_long())
    self$maxTime <- max(dataCat$timeId)
    close(progressBar)
    delta <- Sys.time() - start
    self$numTemporalFeatures <- length(temporalFeatures)
    ParallelLogger::logInfo("Processing temporal features for dataset took ", signif(delta, 3), " ", attr(delta, "units"))
  },
  processStatic = function(data) {
    # create a dense matrix of each static feature
    # these features will then be concatenated to the embeddings
    # later more advanced options than concatenation should be explored
    if (is.null(self$datasetInfo$staticFeatures)) {
      uniqueFeatures <- data %>% 
        dplyr::distinct(covariateId) %>% 
        dplyr::collect() %>% 
        dplyr::pull()
      uniqueFeaturesCount <- length(uniqueFeatures)
      map <- data.frame(covariateId = uniqueFeatures, 
                        newColumnId = 1:uniqueFeaturesCount)
      # remap columns to new matrix (rowIds should stay the same)
      data <- data %>% dplyr::arrange(columnId, rowId)
      data <- data %>% 
        dplyr::inner_join(map, by = "covariateId") %>% 
        dplyr::collect()
      self$datasetInfo$staticFeatures <- list(features = uniqueFeatures,
                                              map = map)
    } else {
      uniqueFeatures <- self$datasetInfo$staticFeatures$features
      uniqueFeaturesCount <- length(uniqueFeatures)
      # make sure to select same features in same columns as during developement
      map <- self$datasetInfo$staticFeatures$map
      data <- data %>% 
        dplyr::inner_join(map, by = "covariateId") %>% 
        dplyr::collect()
      # what if test set has more/less non-temporal features? 
    }
    self$numStaticFeatures <- uniqueFeaturesCount
    self$staticFeatures <- uniqueFeatures
    
    # transform age
    ageIndex <- data$covariateId == 1002
    if (is.null(self$datasetInfo$normInfo$ageMean)) {
      ageMean <- mean(data[ageIndex, ]$covariateValue)
      self$datasetInfo$normInfo$ageMean <- ageMean
    } else {
      ageMean <- self$datasetInfo$normInfo$ageMean
    }
    
    if (is.null(self$datasetInfo$normInfo$ageStd)) {
      ageStd <- sd(data[ageIndex, ]$covariateValue)
      self$datasetInfo$normInfo$ageStd <- ageStd
    } else {
      ageStd <- self$datasetInfo$normInfo$ageStd
    }
    data[ageIndex, ]$covariateValue <- 
      (data[ageIndex, ]$covariateValue - ageMean) / ageStd
    
    indices <- torch::torch_tensor(cbind(data$rowId, data$newColumnId), 
                                   dtype = torch::torch_long())$t_()
    values <- torch::torch_tensor(data$covariateValue,
      dtype = torch::torch_float32())
    self$staticData <- torch::torch_sparse_coo_tensor(
      indices = indices,
      values = values,
      size = c(self$target$shape, self$numStaticFeatures)
    )
  },
  
  processTarget = function(labels, data) {
    # add labels if training (make 0 vector for prediction)
    if (!is.null(labels)) {
      target <- torch::torch_tensor(labels)
    } else {
      target <- torch::torch_tensor(rep(0, data %>% 
        dplyr::distinct(rowId) %>% 
        dplyr::collect() %>% 
        nrow()))
    }
    self$target <- target
  },
  
  # these are parameters from the data required to build model
  getDatasetParams = function() {
    return(list(maxVisit = self$maxVisit,
                temporalFeatures = self$numTemporalFeatures,
                staticFeatures = self$numStaticFeatures,
                maxTime = self$maxTime))
  },
  .getbatch = function(item) {
    batch <- list(
      sequences = torch::torch_stack(self$sequences[item]),
      static = self$staticData$index_select(item, dim = 1),
      lengths = self$lengths[item],
      visits = torch::torch_stack(self$visits[item])
    )
    return(list(
      batch = batch,
      target = self$target[item]
    ))
  },
  .length = function() {
    self$target$size()[[1]] 
  }
)
