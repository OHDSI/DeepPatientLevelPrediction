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
#' @param ... 
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
  
  mappedData <- PatientLevelPredictionArrow::toSparseM(
    plpData = trainData,  
    map = NULL
  )
  
  matrixData <- mappedData$dataMatrix
  labels <- mappedData$labels
  covariateRef <- mappedData$covariateRef
  
  outLoc <- PatientLevelPredictionArrow:::createTempModelLoc() # export
  
  cvResult <- do.call( 
    what = gridCvDeep,
    args = list(
      matrixData = matrixData,
      labels = labels,
      settings = settings,
      modelLocation = outLoc,
      paramSearch = param
    )
  )
  
  hyperSummary <- do.call(rbind, lapply(cvResult$paramGridSearch, function(x) x$hyperSummary))
  
  prediction <- cvResult$prediction
  
  incs <- rep(1, nrow(covariateRef))
  covariateRef$included <- incs
  covariateRef$covariateValue <- 0
  
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
    
    dataMat <- toSparseM(
      plpData = data, 
      cohort = cohort, 
      map = plpModel$covariateImportance %>% 
        dplyr::select(.data$columnId, .data$covariateId)
    )
    data <- Dataset(dataMat$dataMatrix) # add numeric details..
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
    prediction$value <- estimator$predictProba(data)
  } else {
    prediction$value <- plpModel$model$predictProba(data)
  }
  
  
  attr(prediction, "metaData")$modelType <-  attr(plpModel, 'modelType')
  
  return(prediction)
}

#' gr idCvDeep 
#' 
#' @description 
#' Performs grid search for a deep learning estimator
#' 
#' 
#' @param matrixData    Data in sparse matrix format
#' @param labels        Dataframe with the outcomes
#' @param settings      Settings of the model
#' @param modelLocation Where to save the model
#' @param paramSearch   model parameters to perform search over
#' 
#' @export 
gridCvDeep <- function(
  matrixData,
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
  dataset <- Dataset(matrixData, labels$outcomeCount)
  
  
  for(gridId in 1:length(paramSearch)){
    
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
      
      browser()
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
      
    }
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
  fitParams$epochs <- epochs
  fitParams$batchSize <- batchSize
  # create the dir
  if(!dir.exists(file.path(modelLocation))){
    dir.create(file.path(modelLocation), recursive = T)
  }
  trainDataset <- Dataset(
    matrixData, 
    labels$outcomeCount
  )
  modelParams$catFeatures <- trainDataset$numCatFeatures()
  modelParams$numFeatures <- trainDataset$numNumFeatures()
  
  
  estimator <- Estimator$new(
    baseModel = baseModel,
    modelParameters = modelParams,
    fitParameters = fitParams, 
    device = device
  )
  numericalIndex <- trainDataset$.getNumericalIndex()
  
  estimator$fitWholeTrainingSet(trainDataset)
  
  
  ParallelLogger::logInfo("Calculating predictions on all train data...")
  prediction <- predictDeepEstimator(
    plpModel = estimator, 
    data = trainDataset, 
    cohort = labels
  )
  prediction$evaluationType <- 'Train'
  
  prediction <- rbind(
    prediction,
    cvPrediction
  )
  
  # modify prediction 
  prediction <- prediction %>% 
    dplyr::select(-.data$rowId, -.data$index) %>%
    dplyr::rename(rowId = .data$originalRowId)
  
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

# Estimator
#' @description 
#' A generic R6 class that wraps around a torch nn module and can be used to 
#' fit and predict the model defined in that module.
#' @export
Estimator <- R6::R6Class(
  classname = 'Estimator',
  lock_objects = FALSE,
  public = list(
    initialize = function(baseModel, 
                          modelParameters, 
                          fitParameters,
                          optimizer=torch::optim_adam,
                          criterion=torch::nn_bce_with_logits_loss,
                          device='cpu', 
                          patience=NULL){
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
      
      if (!is.null(patience)) {
        self$earlyStopper <- EarlyStopping$new(patience=patience)
      } else {
        self$earlyStopper <- FALSE
      }
      
      self$bestScore <- NULL
      self$bestEpoch <- NULL
    },
  
    # fits the estimator
    fit = function(dataset, testDataset) {
      valLosses <- c()
      valAUCs <- c()
      
      # dataloader <- torch::dataloader(dataset, 
      #                                 batch_size = self$batchSize, 
      #                                 shuffle = T)
      # testDataloader <- torch::dataloader(testDataset, 
      #                                     batch_size = self$batchSize, 
      #                                     shuffle = F)
      batchIndex <- torch::torch_randperm(length(dataset)) + 1L
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex)/self$batchSize))
      
      testBatchIndex <- 1:length(testDataset)
      testBatchIndex <- split(testBatchIndex, ceiling(seq_along(testBatchIndex)/self$batchSize))
      
      modelStateDict <- list()
      epoch <- list()
      times <- list()
      
      for (epochI in 1:self$epochs) {
        # fit the model
        startTime <- Sys.time()
        self$fitEpoch(dataset, batchIndex)
        endTime <- Sys.time()
        
        # predict on test data
        scores <- self$score(testDataset, testBatchIndex)
        delta <- endTime - startTime
        currentEpoch <- epochI + self$previousEpochs
        ParallelLogger::logInfo('Epochs: ', currentEpoch, 
                                ' | Val AUC: ', round(scores$auc,3), 
                                ' | Val Loss: ', round(scores$loss,3),
                                ' | Time: ', round(delta, 3), ' ', 
                                units(delta))
                                
        valLosses <- c(valLosses, scores$loss)
        valAUCs <- c(valAUCs, scores$auc)
        times <- c(times, round(delta, 3))
        if (self$earlyStopper){
          self$earlyStopper$call(scores$auc)
          if (self$earlyStopper$improved) {
            # here it saves the results to lists rather than files
            modelStateDict[[epochI]]  <- self$model$state_dict()
            epoch[[epochI]] <- currentEpoch
          }
          if (self$earlyStopper$earlyStop) {
            ParallelLogger::logInfo('Early stopping, validation AUC stopped improving')
            ParallelLogger::logInfo('Average time per epoch was: ', mean(as.numeric(times)), ' ' , units(delta))
            self$finishFit(valAUCs, modelStateDict, valLosses, epoch)
            return(invisible(self))
          }
        } else {
          modelStateDict[[epochI]]  <- self$model$state_dict()
          epoch[[epochI]] <- currentEpoch
          }
      }
      ParallelLogger::logInfo('Average time per epoch was: ', mean(as.numeric(times)), ' ' , units(delta))
      self$finishFit(valAUCs, modelStateDict, valLosses, epoch)
      invisible(self)
    },
    
    # trains for one epoch
    fitEpoch = function(dataset, batchIndex){
      self$model$train()
      coro::loop(for (b in batchIndex) {
        self$optimizer$zero_grad()
        cat <- dataset[b]$cat$to(device=self$device)
        num <- dataset[b]$num$to(device=self$device)
        target <- dataset[b]$target$to(device=self$device)
        out <- self$model(num, cat)
        loss <- self$criterion(out, target)
        loss$backward()
        self$optimizer$step()
        })
      
    },
    
    # operations that run when fitting is finished
    finishFit = function(valAUCs, modelStateDict, valLosses, epoch) {
      #extract best epoch from the saved checkpoints
      bestEpochInd <- which.max(valAUCs)  # change this if a different metric is used
      
      bestModelStateDict <- modelStateDict[[bestEpochInd]]
      self$model$load_state_dict(bestModelStateDict)
      
      bestEpoch <- epoch[[bestEpochInd]]
      self$bestEpoch <- bestEpoch
      self$bestScore <- list(loss = valLosses[bestEpochInd], 
                             auc = valAUCs[bestEpochInd])
      
      ParallelLogger::logInfo('Loaded best model (based on AUC) from epoch ', bestEpoch)
      ParallelLogger::logInfo('ValLoss: ', self$bestScore$loss)
      ParallelLogger::logInfo('valAUC: ', self$bestScore$auc)
    },
    
    # Fits whole training set on a specific number of epochs
    # TODO What happens when learning rate changes per epochs?
    # Ideally I would copy the learning rate strategy from before
    # and adjust for different sizes ie more iterations/updates???
    fitWholeTrainingSet = function(dataset) {
      # dataloader <- torch::dataloader(dataset, 
      #                                 batch_size=self$batchSize, 
      #                                 shuffle=TRUE)
      batchIndex <- torch::torch_randperm(length(dataset)) + 1L
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex)/self$batchSize))
      for (epoch in 1:self$epochs) {
        self$fitEpoch(dataset, batchIndex)
      }
      
    }, 
    
    # save model and those parameters needed to reconstruct it
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
    
    # calculates loss and auc after training for one epoch
    score = function(dataset, batchIndex){
      torch::with_no_grad({
        loss = c()
        predictions = c()
        targets = c()
        self$model$eval()
        coro::loop(for (b in batchIndex) {
          b <- self$batchToDevice(dataset[b])
          cat <- b$cat
          num <- b$num
          target <- b$target
          
          pred <- self$model(num, cat)
          predictions <- c(predictions, as.array(pred$cpu()))
          targets <- c(targets, as.array(target$cpu()))
          loss <- c(loss, self$criterion(pred, target)$item())
        })
        mean_loss <- mean(loss)
        predictionsClass <- data.frame(value=predictions, outcomeCount=targets)
        attr(predictionsClass, 'metaData')$predictionType <-'binary' #old can be remvoed
        attr(predictionsClass, 'metaData')$modelType <-'binary' 
        auc <- computeAuc(predictionsClass)
      })
      return(list(loss=mean_loss, auc=auc))
    },
    
    # predicts and outputs the probabilities
    predictProba = function(dataset) {
      # dataloader <- torch::dataloader(dataset, 
      #                                 batch_size = self$batchSize, 
      #                                 shuffle=F)
      batchIndex <- 1:length(dataset)
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex)/self$batchSize))
      torch::with_no_grad({
        predictions <- c()
        self$model$eval()
        coro::loop(for (b in batchIndex){
          b <- self$batchToDevice(dataset[b])
          cat <- b$cat
          num <- b$num
          target <- b$target
          pred <- self$model(num,cat)
          predictions <- c(predictions, as.array(torch::torch_sigmoid(pred$cpu())))
        })
      })
      return(predictions)
    },
    
    
    # predicts and outputs the class
    predict = function(dataset){
      predictions <- self$predict_proba(dataset)
      predicted_class <- torch::torch_argmax(torch::torch_unsqueeze(torch::torch_tensor(predictions), dim=2),dim=2)
      return(predicted_class)
    },
    
    # sends a batch of data to device
    ## TODO make agnostic of the form of batch
    batchToDevice = function(batch) {
      cat <- batch[[1]]$to(device=self$device)
      num <- batch[[2]]$to(device=self$device)
      target <- batch[[3]]$to(device=self$device)
      
      result <- list(cat=cat, num=num, target=target)
      return(result)
    },
    
    # select item from list, and if it's null sets a default
    itemOrDefaults = function (list, item, default = NULL) {
      value <- list[[item]]
      if (is.null(value)) default else value
    }
  )
)

EarlyStopping <- R6::R6Class(
   classname = 'EarlyStopping',
   lock_objects = FALSE,
   public = list(
     initialize = function(patience=3, delta=0) {
       self$patience <- patience
       self$counter <- 0
       self$bestScore <- NULL
       self$earlyStop <- FALSE
       self$improved <- FALSE
       self$delta <- delta
       self$previousScore <- 0
     },
     call = function(metric){
       score <- metric
       if (is.null(self$bestScore)) {
         self$bestScore <- score
         self$improved <- TRUE
       }
       else if (score < self$bestScore + self$delta) {
         self$counter <- self$counter + 1
         self$improved <- FALSE
         ParallelLogger::logInfo('EarlyStopping counter: ', self$counter,
                                 ' out of ', self$patience)
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


