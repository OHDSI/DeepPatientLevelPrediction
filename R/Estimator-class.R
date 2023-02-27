# @file Estimator-class.R
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

#' Estimator
#' @description
#' A generic R6 class that wraps around a torch nn module and can be used to
#' fit and predict the model defined in that module.
#' @export
Estimator <- R6::R6Class(
  classname = "Estimator",
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new estimator
    #' @param modelType The torch nn module to use as model
    #' @param modelParameters Parameters to initialize the model
    #' @param estimatorSettings Parameters required for the estimator fitting
    initialize = function(modelType,
                          modelParameters,
                          estimatorSettings) {
      self$seed <- estimatorSettings$seed
      self$device <- estimatorSettings$device
      torch::torch_manual_seed(seed=self$seed)
      self$model <- do.call(modelType, modelParameters)
      self$modelParameters <- modelParameters
      self$estimatorSettings <- estimatorSettings
      self$epochs <- self$itemOrDefaults(estimatorSettings, "epochs", 10)
      self$learningRate <- self$itemOrDefaults(estimatorSettings, "learningRate", 1e-3)
      self$l2Norm <- self$itemOrDefaults(estimatorSettings, "weightDecay", 1e-5)
      self$batchSize <- self$itemOrDefaults(estimatorSettings, "batchSize", 1024)
      self$prefix <- self$itemOrDefaults(estimatorSettings, "prefix", self$model$name)
      
      self$previousEpochs <- self$itemOrDefaults(estimatorSettings, "previousEpochs", 0)
      self$model$to(device = self$device)
      
      self$optimizer <- estimatorSettings$optimizer(
        params = self$model$parameters,
        lr = self$learningRate,
        weight_decay = self$l2Norm
      )
      self$criterion <- estimatorSettings$criterion()
      
      if (!is.null(estimatorSettings$metric)) {
        self$metric <- estimatorSettings$metric
        if (is.character(self$metric)) {
          if (self$metric == "auc") {
            self$metric <- list(name="auc",
                                mode="max")
          } else if (self$metric == "loss") {
            self$metric <- list(name="loss",
                                mode="min")
          }
        }
        if (!is.null(estimatorSettings$scheduler)) {
          estimatorSettings$scheduler$params$mode <- self$metrix$mode 
        }
        if (!is.null(estimatorSettings$earlyStopping)) {
          estimatorSettings$earlyStopping$params$mode <- self$metric$mode
        }
      }
      
      if (!is.null(estimatorSettings$scheduler)) {
        self$scheduler <- do.call(estimatorSettings$scheduler$fun,
                                  c(self$optimizer, estimatorSettings$scheduler$params))
      }
      
      # gradient accumulation is useful when training large numbers where
      # you can only fit few samples on the GPU in each batch.
      self$gradAccumulationIter <- 1
      
      if (!is.null(estimatorSettings$earlyStopping) && estimatorSettings$earlyStopping$useEarlyStopping) {
        self$earlyStopper <- do.call(EarlyStopping$new, estimatorSettings$earlyStopping$params)
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
      allScores <- list()
      batchIndex <- torch::torch_randperm(length(dataset)) + 1L
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex) / self$batchSize))
      
      testBatchIndex <- 1:length(testDataset)
      testBatchIndex <- split(testBatchIndex, ceiling(seq_along(testBatchIndex) / self$batchSize))
      
      modelStateDict <- list()
      epoch <- list()
      times <- list()
      learnRates <- list()
      for (epochI in 1:self$epochs) {
        startTime <- Sys.time()
        trainLoss <- self$fitEpoch(dataset, batchIndex)
        endTime <- Sys.time()
        
        # predict on test data
        scores <- self$score(testDataset, testBatchIndex)
        delta <- endTime - startTime
        currentEpoch <- epochI + self$previousEpochs
        lr <- self$optimizer$param_groups[[1]]$lr
        self$printProgress(scores, trainLoss, delta, currentEpoch)
        self$scheduler$step(scores$metric)
        allScores[[epochI]] <- scores
        learnRates <- c(learnRates, lr)
        times <- c(times, round(delta, 3))
        if (!is.null(self$earlyStopper)) {
          self$earlyStopper$call(scores$metric)
          if (self$earlyStopper$improved) {
            # here it saves the results to lists rather than files
            modelStateDict[[epochI]] <- lapply(self$model$state_dict(), function(x) x$detach()$cpu())
            epoch[[epochI]] <- currentEpoch
          }
          if (self$earlyStopper$earlyStop) {
            ParallelLogger::logInfo("Early stopping, validation metric stopped improving")
            ParallelLogger::logInfo("Average time per epoch was: ", round(mean(as.numeric(times)), 3), " ", units(delta))
            self$finishFit(allScores, modelStateDict, epoch, learnRates)
            return(invisible(self))
          }
        } else {
          modelStateDict[[epochI]] <- lapply(self$model$state_dict(), function(x) x$detach()$cpu())
          epoch[[epochI]] <- currentEpoch
        }
      }
      ParallelLogger::logInfo("Average time per epoch was: ", round(mean(as.numeric(times)), 3), " ", units(delta))
      self$finishFit(allScores, modelStateDict, epoch, learnRates)
      invisible(self)
    },
    
    #' @description
    #' fits estimator for one epoch (one round through the data)
    #' @param dataset     torch dataset to use for fitting
    #' @param batchIndex  indices of batches
    fitEpoch = function(dataset, batchIndex) {
      trainLosses <- torch::torch_empty(length(batchIndex))
      ix <- 1
      self$model$train()
      progressBar <- utils::txtProgressBar(style = 3)
      for (b in batchIndex) {
        self$optimizer$zero_grad()
        batch <- batchToDevice(dataset[b], device=self$device)
        out <- self$model(batch[[1]])
        loss <- self$criterion(out, batch[[2]])
        loss$backward()
        
        self$optimizer$step()
        trainLosses[ix] <- loss$detach()
        utils::setTxtProgressBar(progressBar, ix / length(batchIndex))
        ix <- ix + 1
      }
      close(progressBar)
      trainLosses$mean()$item()
    },
    
    #' @description
    #' calculates loss and auc after training for one epoch
    #' @param dataset    The torch dataset to use to evaluate loss and auc
    #' @param batchIndex Indices of batches in the dataset
    #' @return list with average loss and auc in the dataset
    score = function(dataset, batchIndex) {
      torch::with_no_grad({
        loss <- torch::torch_empty(c(length(batchIndex)))
        predictions <- list()
        targets <- list()
        self$model$eval()
        ix <- 1
        for (b in batchIndex) {
          batch <- batchToDevice(dataset[b], device=self$device)
          pred <- self$model(batch[[1]])
          predictions <- c(predictions, pred)
          targets <- c(targets, batch[[2]])
          loss[ix] <- self$criterion(pred, batch[[2]])
          ix <- ix + 1
        }
        mean_loss <- loss$mean()$item()
        predictionsClass <- data.frame(
          value = as.matrix(torch::torch_sigmoid(torch::torch_cat(predictions)$cpu())),
          outcomeCount = as.matrix(torch::torch_cat(targets)$cpu())
        )
        attr(predictionsClass, "metaData")$modelType <- "binary"
        auc <- PatientLevelPrediction::computeAuc(predictionsClass)
        scores <- list()
        if (!is.null(self$metric)) {
          if (self$metric$name == "auc") {
            scores$metric <- auc
          } else if (self$metric$name == "loss") {
            scores$metric <- mean_loss
          } else {
            metric <- self$metric$fun(predictionsClass$value, predictionsClass$outcomeCount)
            scores$metric <- metric
          }
        }
        scores$auc <- auc
        scores$loss <- mean_loss
      })
      return(scores)
    },
    
    #' @description
    #' operations that run when fitting is finished
    #' @param scores          validation scores
    #' @param modelStateDict  fitted model parameters
    #' @param epoch           list of epochs fit
    #' @param learnRates      learning rate sequence used so far
    finishFit = function(scores, modelStateDict, epoch, learnRates) {
      if (self$metric$mode=="max") {
        bestEpochInd <- which.max(unlist(lapply(scores, function(x) x$metric)))
      }
      else if (self$metric$mode=="min") {
        bestEpochInd <- which.min(unlist(lapply(scores, function(x) x$metric)))
      }
      
      bestModelStateDict <- lapply(modelStateDict[[bestEpochInd]], function(x) x$to(device = self$device))
      self$model$load_state_dict(bestModelStateDict)
      
      bestEpoch <- epoch[[bestEpochInd]]
      self$bestEpoch <- bestEpoch
      self$bestScore <- list(
        loss = scores[[bestEpochInd]]$loss,
        auc = scores[[bestEpochInd]]$auc
      )
      self$learnRateSchedule <- learnRates[1:bestEpochInd]
      
      ParallelLogger::logInfo("Loaded best model (based on AUC) from epoch ", bestEpoch)
      ParallelLogger::logInfo("ValLoss: ", self$bestScore$loss)
      ParallelLogger::logInfo("valAUC: ", self$bestScore$auc)
      if (!is.null(self$metric) && (!self$metric$name=='auc') && (!self$metric$name=='loss')) {
        self$bestScore[[self$metric$name]] <- scores[[bestEpochInd]]$metric
        ParallelLogger::logInfo(self$metric$name,": ", self$bestScore[[self$metric$name]])
      }
    },
    
    #' @description Print out training progress per epoch
    #' @param scores scores returned by `self$score`
    #' @param trainLoss training loss
    #' @param delta how long did the epoch take
    #' @param currentEpoch the current epoch number
    printProgress = function(scores, trainLoss, delta, currentEpoch) {
      if (!is.null(self$metric) && (!self$metric$name=='auc') && (!self$metric$name=='loss')) {
        ParallelLogger::logInfo(
          "Epochs: ", currentEpoch,
          " | Val ", self$metric$name, ": ", round(scores$metric, 3),
          " | Val AUC: ", round(scores$auc, 3),
          " | Val Loss: ", round(scores$loss, 3),
          " | Train Loss: ", round(trainLoss, 3),
          " | Time: ", round(delta, 3), " ",
          units(delta),
          " | LR: ", self$optimizer$param_groups[[1]]$lr
        )
      } else {
      ParallelLogger::logInfo(
        "Epochs: ", currentEpoch,
        " | Val AUC: ", round(scores$auc, 3),
        " | Val Loss: ", round(scores$loss, 3),
        " | Train Loss: ", round(trainLoss, 3),
        " | Time: ", round(delta, 3), " ",
        units(delta),
        " | LR: ", self$optimizer$param_groups[[1]]$lr
      )
      }
    },
    
    #' @description
    #' Fits whole training set on a specific number of epochs
    #' @param dataset torch dataset
    #' @param learnRates learnRateSchedule from CV
    fitWholeTrainingSet = function(dataset, learnRates = NULL) {
      if (length(learnRates) > 1) {
        self$bestEpoch <- length(learnRates)
      } else if (is.null(self$bestEpoch)) {
        self$bestEpoch <- self$epochs
      }
      # TODO constant LR

      batchIndex <- torch::torch_randperm(length(dataset)) + 1L
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex) / self$batchSize))
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
      torch::torch_save(
        list(
          modelStateDict = self$model$state_dict(),
          modelParameters = self$modelParameters,
          estimatorSettings = self$estimatorSettings,
          epoch = self$epochs
        ),
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
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex) / self$batchSize))
      torch::with_no_grad({
        predictions <- torch::torch_empty(length(dataset), device=self$device)
        self$model$eval()
        progressBar <- utils::txtProgressBar(style = 3)
        ix <- 1
        coro::loop(for (b in batchIndex) {
          batch <- batchToDevice(dataset[b], self$device)
          target <- batch$target
          pred <- self$model(batch$batch)
          predictions[b] <- torch::torch_sigmoid(pred)
          utils::setTxtProgressBar(progressBar, ix / length(batchIndex))
          ix <- ix + 1
        })
        predictions <- as.array(predictions$cpu())
        close(progressBar)
      })
      return(predictions)
    },
    
    #' @description
    #' predicts and outputs the class
    #' @param   dataset A torch dataset to create predictions for
    #' @param   threshold Which threshold to use for predictions
    #' @return  The predicted class for the data in the dataset
    predict = function(dataset, threshold = NULL) {
      predictions <- self$predictProba(dataset)
      
      if (is.null(threshold)) {
        # use outcome rate
        threshold <- dataset$target$sum()$item() / length(dataset)
      }
      predicted_class <- as.integer(predictions > threshold)
      return(predicted_class)
    },
    
    #' @description
    #' select item from list, and if it's null sets a default
    #' @param list A list with items
    #' @param item Which list item to retrieve
    #' @param default The value to return if list doesn't have item
    #' @return the list item or default
    itemOrDefaults = function(list, item, default = NULL) {
      value <- list[[item]]
      if (is.null(value)) default else value
    }
  )
)

#' Earlystopping class
#' @description
#' Stops training if a loss or metric has stopped improving
EarlyStopping <- R6::R6Class(
  classname = "EarlyStopping",
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new earlyStopping object
    #' @param patience Stop after this number of epochs if loss doesn't improve
    #' @param delta    How much does the loss need to improve to count as improvement
    #' @param verbose  If information should be printed out
    #' @param mode    either `min` or `max` depending on metric to be used for earlyStopping
    #' @return a new earlystopping object
    initialize = function(patience = 3, delta = 0, verbose = TRUE,
                          mode='max') {
      self$patience <- patience
      self$counter <- 0
      self$verbose <- verbose
      self$bestScore <- NULL
      self$earlyStop <- FALSE
      self$improved <- FALSE
      self$delta <- delta
      self$previousScore <- 0
      self$mode <- mode
    },
    #' @description
    #' call the earlystopping object and increment a counter if loss is not
    #' improving
    #' @param metric the current metric value
    call = function(metric) {
      if (self$mode=='max') {
        score <- metric
      } else {
        score <- -1 * metric
      }
      if (is.null(self$bestScore)) {
        self$bestScore <- score
        self$improved <- TRUE
      } else if (score < self$bestScore + self$delta) {
        self$counter <- self$counter + 1
        self$improved <- FALSE
        if (self$verbose) {
          ParallelLogger::logInfo(
            "EarlyStopping counter: ", self$counter,
            " out of ", self$patience
          )
        }
        if (self$counter >= self$patience) {
          self$earlyStop <- TRUE
        }
      } else {
        self$bestScore <- score
        self$counter <- 0
        self$improved <- TRUE
      }
      self$previousScore <- score
    }
  )
)

#' sends a batch of data to device
#' @description 
#' sends a batch of data to device
#' assumes batch includes lists of tensors to arbitrary nested depths
#' @param batch the batch to send, usually a list of torch tensors
#' @param device which device to send batch to
#' @return the batch on the required device
batchToDevice = function(batch, device) {
  if (class(batch)[1] == "torch_tensor") {
    batch <- batch$to(device = device)
  } else {
    ix <- 1
    for (b in batch) {
      if (class(b)[1] == "torch_tensor") {
        b <- b$to(device = device)
      } else {
        b <- batchToDevice(b, device)
      }
      if (!is.null(b)) {
        batch[[ix]] <- b
      }
      ix <- ix + 1
    }
  }
  return(batch)
}
