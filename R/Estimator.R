# @file Estimator.R
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

#' fitEstimator
#'
#' @description 
#' fits a deep learning estimator to data.
#' 
#' @param trainData      the data to use
#' @param param          model parameters
#' @param analysisId     Id of the analysis
#' @param ...            Extra inputs
#'
#' @export
fitEstimator <- function(
  trainData, 
  param, 
  analysisId,
  ...
) {
  
  start <- Sys.time()
  
  # check covariate data
  if(!FeatureExtraction::isCovariateData(trainData$covariateData)){stop("Needs correct covariateData")}
  
  # get the settings from the param
  settings <- attr(param, 'settings')
  if(!is.null(trainData$folds)){
    trainData$labels <- merge(trainData$labels, trainData$fold, by = 'rowId')
  }
  mappedCovariateData <- PatientLevelPrediction:::MapIds(covariateData = trainData$covariateData,
                                                         cohort = trainData$labels)
  
  covariateRef <- mappedCovariateData$covariateRef
  
  outLoc <- PatientLevelPrediction:::createTempModelLoc() # export
  cvResult <- do.call( 
    what = gridCvDeep,
    args = list(
      mappedData = mappedCovariateData,
      labels = trainData$labels,
      settings = settings,
      modelLocation = outLoc,
      paramSearch = param
    )
  )
  
  hyperSummary <- do.call(rbind, lapply(cvResult$paramGridSearch, function(x) x$hyperSummary))
  prediction <- cvResult$prediction
  incs <- rep(1, covariateRef %>% dplyr::tally() %>% dplyr::pull())
  covariateRef <- covariateRef %>% dplyr::collect() %>% 
                                   dplyr::mutate(included=incs,
                                                 covariateValue=0)
                  
  
  comp <- start - Sys.time()
  result <- list(
    model = cvResult$estimator, #file.path(outLoc),
    
    prediction = prediction,
    
    settings = list(
      plpDataSettings = attr(trainData, "metaData")$plpDataSettings,
      covariateSettings = attr(trainData, "metaData")$covariateSettings,
      populationSettings = attr(trainData, "metaData")$populationSettings,
      featureEngineering = attr(trainData$covariateData, "metaData")$featureEngineering,
      tidyCovariates = attr(trainData$covariateData, "metaData")$tidyCovariateDataSettings, 
      requireDenseMatrix = F,
      modelSettings = list(
        model = settings$name, 
        param = param,
        finalModelParameters = cvResult$finalParam,
        numericalIndex = cvResult$numericalIndex,
        extraSettings = attr(param, 'settings')
      ),
      splitSettings = attr(trainData, "metaData")$splitSettings,
      sampleSettings = attr(trainData, "metaData")$sampleSettings
    ),
    
    trainDetails = list(
      analysisId = analysisId,
      cdmDatabaseSchema = attr(trainData, "metaData")$cdmDatabaseSchema,
      outcomeId = attr(trainData, "metaData")$outcomeId,
      cohortId = attr(trainData, "metaData")$cohortId,
      attrition = attr(trainData, "metaData")$attrition, 
      trainingTime = comp,
      trainingDate = Sys.Date(),
      hyperParamSearch = hyperSummary
    ),
    
    covariateImportance = covariateRef
  )
  
  class(result) <- "plpModel"
  attr(result, "predictionFunction") <- "predictDeepEstimator"
  attr(result, "modelType") <- "binary"
  attr(result, "saveType") <- attr(param, 'settings')$saveType
  
  return(result)
}

#' predictDeepEstimator
#'
#' @description 
#' the prediction function for the estimator
#' 
#' @param plpModel   the plpModel
#' @param data       plp data object or a torch dataset
#' @param cohort     data.frame with the rowIds of the people
#'
#' @export 
predictDeepEstimator <- function(
  plpModel, 
  data, 
  cohort
){
  if(!'plpModel' %in% class(plpModel)){
    plpModel <- list(model = plpModel)
    attr(plpModel, 'modelType') <- 'binary'
  }
  if("plpData" %in% class(data)){
    mappedData <- PatientLevelPrediction:::MapIds(data$covariateData,
                                                  cohort=cohort,
                                                  mapping=plpModel$covariateImportance %>% 
                                                    dplyr::select(.data$columnId, 
                                                                  .data$covariateId))
    data <- Dataset(mappedData$covariates,
                    numericalIndex = plpModel$settings$modelSettings$numericalIndex)
  }
  
  # get predictions
  prediction <- cohort
  
  if(is.character(plpModel$model)){
    model <- torch::torch_load(file.path(plpModel$model, 'DeepEstimatorModel.pt'), device='cpu')
    estimator <- Estimator$new(
      baseModel = plpModel$settings$modelSettings$model,
      modelParameters = model$modelParameters,
      fitParameters = model$fitParameters, 
      device = plpModel$settings$modelSettings$extraSettings$device
    )
    estimator$model$load_state_dict(model$modelStateDict)
    prediction$value <- estimator$predictProba(data)
  } else {
    prediction$value <- plpModel$model$predictProba(data)
  }
  
  
  attr(prediction, "metaData")$modelType <-  attr(plpModel, 'modelType')
  
  return(prediction)
}

#' gridCvDeep 
#' 
#' @description 
#' Performs grid search for a deep learning estimator
#' 
#' 
#' @param mappedData    Mapped data with covariates
#' @param labels        Dataframe with the outcomes
#' @param settings      Settings of the model
#' @param modelLocation Where to save the model
#' @param paramSearch   model parameters to perform search over
#' 
#' @export 
gridCvDeep <- function(
  mappedData,
  labels,
  settings,
  modelLocation,
  paramSearch
){
  
  modelName <- settings$modelName
  modelParamNames <- settings$modelParamNames
  fitParamNames <- c("weightDecay", "learningRate")
  epochs <- settings$epochs
  batchSize <- settings$batchSize
  baseModel <- settings$baseModel
  device <- settings$device
  
  ParallelLogger::logInfo(paste0("Running CV for ",modelName," model"))

  ###########################################################################
  

  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- length(paramSearch)
  dataset <- Dataset(mappedData$covariates, labels$outcomeCount)
  for(gridId in 1:length(paramSearch)){
    ParallelLogger::logInfo(paste0("Running hyperparameter combination no ",gridId))
    ParallelLogger::logInfo(paste0("HyperParameters: "))
    ParallelLogger::logInfo(paste(names(paramSearch[[gridId]]), paramSearch[[gridId]], collapse=' | '))
    modelParams <- paramSearch[[gridId]][modelParamNames]
    
    fitParams <- paramSearch[[gridId]][fitParamNames]
    fitParams$epochs <- epochs
    fitParams$batchSize <- batchSize
    
    
    # initiate prediction
    prediction <- c()
    
    fold <- labels$index
    ParallelLogger::logInfo(paste0('Max fold: ', max(fold)))
    modelParams$catFeatures <- dataset$numCatFeatures()
    modelParams$numFeatures <- dataset$numNumFeatures()
    learnRates <- list()
    for( i in 1:max(fold)){
      ParallelLogger::logInfo(paste0('Fold ',i))
      trainDataset <- torch::dataset_subset(dataset, indices=which(fold!=i)) 
      testDataset <- torch::dataset_subset(dataset, indices=which(fold==i))
      fitParams['posWeight'] <- trainDataset$posWeight
      estimator <- Estimator$new(
        baseModel = baseModel, 
        modelParameters = modelParams,
        fitParameters = fitParams, 
        device = device
      )
      
      estimator$fit(
        trainDataset, 
        testDataset
      )
      
      ParallelLogger::logInfo("Calculating predictions on left out fold set...")
      
      prediction <- rbind(
        prediction, 
        predictDeepEstimator(
          plpModel = estimator, 
          data = testDataset, 
          cohort = labels[fold == i,]
        )
      )
      learnRates[[i]] <- list(LRs=estimator$learnRateSchedule,
                         bestEpoch=estimator$bestEpoch)
    }
    maxIndex <- which.max(unlist(sapply(learnRates, `[`, 2)))
    paramSearch[[gridId]]$learnSchedule <- learnRates[[maxIndex]]
    
    gridSearchPredictons[[gridId]] <- list(
      prediction = prediction,
      param = paramSearch[[gridId]]
      )
  }
  # get best para (this could be modified to enable any metric instead of AUC, just need metric input in function)
  paramGridSearch <- lapply(gridSearchPredictons, function(x){do.call(computeGridPerformance, x)})  # cvAUCmean, cvAUC, param
  
  optimalParamInd <- which.max(unlist(lapply(paramGridSearch, function(x) x$cvPerformance)))
  finalParam <- paramGridSearch[[optimalParamInd]]$param
  
  cvPrediction <- gridSearchPredictons[[optimalParamInd]]$prediction
  cvPrediction$evaluationType <- 'CV'
  
  ParallelLogger::logInfo('Training final model using optimal parameters')
  
  # get the params
  modelParams <- finalParam[modelParamNames]
  fitParams <- finalParam[fitParamNames]
  fitParams$epochs <- finalParam$learnSchedule$bestEpoch
  fitParams$batchSize <- batchSize
  # create the dir
  if(!dir.exists(file.path(modelLocation))){
    dir.create(file.path(modelLocation), recursive = T)
  }
  modelParams$catFeatures <- dataset$numCatFeatures()
  modelParams$numFeatures <- dataset$numNumFeatures()
  
  estimator <- Estimator$new(
    baseModel = baseModel,
    modelParameters = modelParams,
    fitParameters = fitParams, 
    device = device
  )
  numericalIndex <- dataset$getNumericalIndex()
  
  estimator$fitWholeTrainingSet(dataset, finalParam$learnSchedule$LRs)
  
  ParallelLogger::logInfo("Calculating predictions on all train data...")
  prediction <- predictDeepEstimator(
    plpModel = estimator, 
    data = dataset, 
    cohort = labels
  )
  prediction$evaluationType <- 'Train'
  
  prediction <- rbind(
    prediction,
    cvPrediction
  )
  # modify prediction 
  prediction <- prediction %>% 
    dplyr::select(-.data$index)

  prediction$cohortStartDate <- as.Date(prediction$cohortStartDate, origin = '1970-01-01')
  
  
  # save torch code here
  estimatorFile <- estimator$save(modelLocation, 'DeepEstimatorModel.pt')
  
  return(
    list( 
      estimator = modelLocation,
      prediction = prediction,
      finalParam = finalParam,
      paramGridSearch = paramGridSearch,
      numericalIndex = numericalIndex
    )
  )
  
}

#' Estimator
#' @description 
#' A generic R6 class that wraps around a torch nn module and can be used to 
#' fit and predict the model defined in that module.
#' @export
Estimator <- R6::R6Class(
  classname = 'Estimator',
  lock_objects = FALSE,
  public = list(
    #' @description 
    #' Creates a new estimator
    #' @param baseModel       The torch nn module to use as model
    #' @param modelParameters Parameters to initialize the baseModel
    #' @param fitParameters   Parameters required for the estimator fitting
    #' @param optimizer       A torch optimizer to use, default is Adam
    #' @param criterion       The torch loss function to use, defaults to 
    #'                        binary cross entropy with logits
    #' @param scheduler       learning rate scheduler to use                  
    #' @param device           Which device to use for fitting, default is cpu
    #' @param patience         Patience to use for early stopping                      
    initialize = function(baseModel, 
                          modelParameters, 
                          fitParameters,
                          optimizer=torch::optim_adam,
                          criterion=torch::nn_bce_with_logits_loss,
                          scheduler=torch::lr_reduce_on_plateau,
                          device='cpu', 
                          patience=4) {
      self$device <- device
      self$model <- do.call(baseModel, modelParameters)
      self$modelParameters <- modelParameters
      self$fitParameters <- fitParameters
      self$epochs <- self$itemOrDefaults(fitParameters, 'epochs', 10)
      self$learningRate <- self$itemOrDefaults(fitParameters,'learningRate', 1e-3)
      self$l2Norm <- self$itemOrDefaults(fitParameters, 'weightDecay', 1e-5)
      self$batchSize <- self$itemOrDefaults(fitParameters, 'batchSize', 1024)
      self$posWeight <- self$itemOrDefaults(fitParameters, 'posWeight', 1)
      
      self$prefix <- self$itemOrDefaults(fitParameters, 'prefix', self$model$name)
      
      self$previousEpochs <- self$itemOrDefaults(fitParameters, 'previousEpochs', 0)
      self$model$to(device=self$device)
      
      self$optimizer <- optimizer(params=self$model$parameters, 
                                  lr=self$learningRate, 
                                  weight_decay=self$l2Norm)
      self$criterion <- criterion(torch::torch_tensor(self$posWeight, 
                                                      device=self$device))
      
      self$scheduler <- scheduler(self$optimizer, patience=1,
                                  verbose=FALSE, mode='max')
      
      # gradient accumulation is useful when training large numbers where
      # you can only fit few samples on the GPU in each batch.
      self$gradAccumulationIter <- 1
        
      if (!is.null(patience)) {
        self$earlyStopper <- EarlyStopping$new(patience=patience)
      } else {
        self$earlyStopper <- NULL
      }
      
      self$bestScore <- NULL 
      self$bestEpoch <- NULL
    },
  
    #' @description fits the estimator
    #' @param dataset     a torch dataset to use for model fitting
    #' @param testDataset a torch dataset to use for early stopping
    fit = function(dataset, testDataset) {
      valLosses <- c()
      valAUCs <- c()
      batchIndex <- torch::torch_randperm(length(dataset)) + 1L
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex)/self$batchSize))

      testBatchIndex <- 1:length(testDataset)
      testBatchIndex <- split(testBatchIndex, ceiling(seq_along(testBatchIndex)/self$batchSize))

      modelStateDict <- list()
      epoch <- list()
      times <- list()
      learnRates <-list()
      for (epochI in 1:self$epochs) {
        startTime <- Sys.time()
        trainLoss <- self$fitEpoch(dataset, batchIndex)
        endTime <- Sys.time()
        
        # predict on test data
        scores <- self$score(testDataset, testBatchIndex)
        delta <- endTime - startTime
        currentEpoch <- epochI + self$previousEpochs
        lr <- self$optimizer$param_groups[[1]]$lr
        ParallelLogger::logInfo('Epochs: ', currentEpoch, 
                                ' | Val AUC: ', round(scores$auc,3), 
                                ' | Val Loss: ', round(scores$loss,3),
                                ' | Train Loss: ', round(trainLoss,3),
                                ' | Time: ', round(delta, 3), ' ', 
                                units(delta),
                                ' | LR: ', lr)
        self$scheduler$step(scores$auc)                        
        valLosses <- c(valLosses, scores$loss)
        valAUCs <- c(valAUCs, scores$auc)
        learnRates <- c(learnRates, lr)
        times <- c(times, round(delta, 3))
        if (!is.null(self$earlyStopper)) {
          self$earlyStopper$call(scores$auc)
          if (self$earlyStopper$improved) {
            # here it saves the results to lists rather than files
            modelStateDict[[epochI]]  <- lapply(self$model$state_dict(), function(x) x$detach()$cpu())
            epoch[[epochI]] <- currentEpoch
          }
          if (self$earlyStopper$earlyStop) {
            ParallelLogger::logInfo('Early stopping, validation AUC stopped improving')
            ParallelLogger::logInfo('Average time per epoch was: ', round(mean(as.numeric(times)),3), ' ' , units(delta))
            self$finishFit(valAUCs, modelStateDict, valLosses, epoch, learnRates)
            return(invisible(self))
          }
        } else {
          modelStateDict[[epochI]]  <- lapply(self$model$state_dict(), function(x) x$detach()$cpu())
          epoch[[epochI]] <- currentEpoch
          }
      }
      ParallelLogger::logInfo('Average time per epoch was: ', round(mean(as.numeric(times)),3), ' ' , units(delta))
      self$finishFit(valAUCs, modelStateDict, valLosses, epoch, learnRates)
      invisible(self)
    },
    
    #' @description 
    #' fits estimator for one epoch (one round through the data)
    #' @param dataset     torch dataset to use for fitting
    #' @param batchIndex  indices of batches 
    fitEpoch = function(dataset, batchIndex){
      trainLosses <- torch::torch_empty(length(batchIndex))
      ix <- 1
      self$model$train()
      progressBar <- utils::txtProgressBar(style=3)
      coro::loop(for (b in batchIndex) {
        self$optimizer$zero_grad()
        batch <- self$batchToDevice(dataset[b])
        out <- self$model(batch[[1]])
        loss <- self$criterion(out, batch[[2]])
        loss$backward()
        
        self$optimizer$step()
        trainLosses[ix] <- loss$detach()
        utils::setTxtProgressBar(progressBar, ix/length(batchIndex))
        ix <- ix + 1
        })
      close(progressBar)
      trainLosses$mean()$item()
    },
    
    #' @description 
    #' calculates loss and auc after training for one epoch
    #' @param dataset    The torch dataset to use to evaluate loss and auc
    #' @param batchIndex Indices of batches in the dataset
    #' @return list with average loss and auc in the dataset
    score = function(dataset, batchIndex){
      torch::with_no_grad({
        loss <- torch::torch_empty(c(length(batchIndex)))
        predictions = list()
        targets = list()
        self$model$eval()
        ix <- 1
        coro::loop(for (b in batchIndex) {
          batch <- self$batchToDevice(dataset[b])
          pred <- self$model(batch[[1]])
          predictions <- c(predictions, pred)
          targets <- c(targets, batch[[2]])
          loss[ix] <- self$criterion(pred, batch[[2]])
          ix <- ix + 1
        })
        mean_loss <- loss$mean()$item()
        predictionsClass <- data.frame(value=as.matrix(torch::torch_sigmoid(torch::torch_cat(predictions)$cpu())), 
                                       outcomeCount=as.matrix(torch::torch_cat(targets)$cpu()))
        attr(predictionsClass, 'metaData')$modelType <-'binary' 
        auc <- PatientLevelPrediction::computeAuc(predictionsClass)
      })
      return(list(loss=mean_loss, auc=auc))
    },
    
    #' @description 
    #' operations that run when fitting is finished
    #' @param valAUCs         validation AUC values
    #' @param modelStateDict  fitted model parameters
    #' @param valLosses       validation losses
    #' @param epoch           list of epochs fit
    #' @param learnRates      learning rate sequence used so far
    finishFit = function(valAUCs, modelStateDict, valLosses, epoch, learnRates) {
      bestEpochInd <- which.max(valAUCs)  # change this if a different metric is used
      
      bestModelStateDict <- lapply(modelStateDict[[bestEpochInd]], function(x) x$to(device=self$device))
      self$model$load_state_dict(bestModelStateDict)
      
      bestEpoch <- epoch[[bestEpochInd]]
      self$bestEpoch <- bestEpoch
      self$bestScore <- list(loss = valLosses[bestEpochInd], 
                             auc = valAUCs[bestEpochInd])
      self$learnRateSchedule <- learnRates[1:bestEpochInd]
      
      ParallelLogger::logInfo('Loaded best model (based on AUC) from epoch ', bestEpoch)
      ParallelLogger::logInfo('ValLoss: ', self$bestScore$loss)
      ParallelLogger::logInfo('valAUC: ', self$bestScore$auc)
    },
    
    #' @description 
    #' Fits whole training set on a specific number of epochs
    #' TODO What happens when learning rate changes per epochs?
    #' Ideally I would copy the learning rate strategy from before
    #' and adjust for different sizes ie more iterations/updates???
    #' @param dataset torch dataset
    #' @param learnRates learnRateSchedule from CV
    fitWholeTrainingSet = function(dataset, learnRates=NULL) {
      if(is.null(self$bestEpoch)) {
        self$bestEpoch <- self$epochs
      }
      
      batchIndex <- torch::torch_randperm(length(dataset)) + 1L
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex)/self$batchSize))
      for (epoch in 1:self$bestEpoch) {
        self$optimizer$param_groups[[1]]$lr <- learnRates[[epoch]]
        self$fitEpoch(dataset, batchIndex)
      }
      
    }, 
    
    #' @description 
    #' save model and those parameters needed to reconstruct it
    #' @param path where to save the model
    #' @param name name of file
    #' @return the path to saved model
    save = function(path, name) {
      savePath <- file.path(path, name)
      torch::torch_save(list(modelStateDict=self$model$state_dict(),
                             modelParameters=self$modelParameters,
                             fitParameters=self$fitParameters,
                             epoch=self$epochs),
                        savePath
                        )
      return(savePath)
      
    },
    
    
    #' @description 
    #' predicts and outputs the probabilities
    #' @param dataset Torch dataset to create predictions for
    #' @return predictions as probabilities
    predictProba = function(dataset) {
      batchIndex <- 1:length(dataset)
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex)/self$batchSize))
      torch::with_no_grad({
        predictions <- c()
        self$model$eval()
        coro::loop(for (b in batchIndex){
          batch <- self$batchToDevice(dataset[b])
          target <- batch$target
          pred <- self$model(batch$batch)
          predictions <- c(predictions, as.array(torch::torch_sigmoid(pred$cpu())))
        })
      })
      return(predictions)
    },
    
    #' @description 
    #' predicts and outputs the class
    #' @param   dataset A torch dataset to create predictions for
    #' @param   threshold Which threshold to use for predictions
    #' @return  The predicted class for the data in the dataset
    predict = function(dataset, threshold=NULL){
      predictions <- self$predictProba(dataset)
      
      if (is.null(threshold)) {
        # use outcome rate
        threshold <- dataset$target$sum()$item()/length(dataset)
      }
      predicted_class <- as.integer(predictions > threshold)
      return(predicted_class)
    },
    
    #' @description 
    #' sends a batch of data to device
    #' assumes batch includes lists of tensors to arbitrary nested depths
    #' @param batch the batch to send, usually a list of torch tensors
    #' @return the batch on the required device
    batchToDevice = function(batch) {
      if (class(batch)[1] == 'torch_tensor') {
        batch <- batch$to(device=self$device)
      } else {
        ix <- 1
        for (b in batch) {
          if (class(b)[1] == 'torch_tensor') {
            b <- b$to(device=self$device)
          } else {
            b <- self$batchToDevice(b)
          }
          if (!is.null(b)) {
            batch[[ix]] <- b
          }
          ix <- ix + 1
        }
      }
      return(batch)
    },
    
    #' @description
    #' select item from list, and if it's null sets a default
    #' @param list A list with items
    #' @param item Which list item to retrieve
    #' @param default The value to return if list doesn't have item
    #' @return the list item or default 
    itemOrDefaults = function (list, item, default = NULL) {
      value <- list[[item]]
      if (is.null(value)) default else value
    }
  )
)

#' Earlystopping class
#' @description 
#' Stops training if a loss or metric has stopped improving
EarlyStopping <- R6::R6Class(
   classname = 'EarlyStopping',
   lock_objects = FALSE,
   public = list(
     #' @description 
     #' Creates a new earlystopping object
     #' @param patience Stop after this number of epochs if loss doesn't improve
     #' @param delta    How much does the loss need to improve to count as improvement
     #' @param verbose   
     #' @return a new earlystopping object
     initialize = function(patience=3, delta=0, verbose=TRUE) {
       self$patience <- patience
       self$counter <- 0
       self$verbose <- verbose
       self$bestScore <- NULL
       self$earlyStop <- FALSE
       self$improved <- FALSE
       self$delta <- delta
       self$previousScore <- 0
     },
     #' @description
     #' call the earlystopping object and increment a counter if loss is not
     #' improving
     #' @param metric the current metric value
     call = function(metric){
       score <- metric
       if (is.null(self$bestScore)) {
         self$bestScore <- score
         self$improved <- TRUE
       }
       else if (score < self$bestScore + self$delta) {
         self$counter <- self$counter + 1
         self$improved <- FALSE
         if (self$verbose) {
         ParallelLogger::logInfo('EarlyStopping counter: ', self$counter,
                                 ' out of ', self$patience)
         }
         if (self$counter >= self$patience) {
           self$earlyStop <- TRUE
         }
       }
       else {
         self$bestScore <- score
         self$counter <- 0
         self$improved <- TRUE
       }
       self$previousScore <- score
     } 
   )
)


