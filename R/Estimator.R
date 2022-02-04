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
      
      dataloader <- torch::dataloader(dataset, 
                                      batch_size = self$batchSize, 
                                      shuffle = T)
      testDataloader <- torch::dataloader(testDataset, 
                                          batch_size = self$batchSize, 
                                          shuffle = F)
      
      modelStateDict <- list()
      epoch <- list()
      times <- list()
      
      for (epochI in 1:self$epochs) {
        
        # fit the model
        startTime <- Sys.time()
        self$fitEpoch(dataloader)
        endTime <- Sys.time()
        
        # predict on test data
        scores <- self$score(testDataloader)
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
    fitEpoch = function(dataloader){
      self$model$train()
      coro::loop(for (b in dataloader) {
        self$optimizer$zero_grad()
        cat <- b[[1]]$to(device=self$device)
        num <- b[[2]]$to(device=self$device)
        target <- b[[3]]$to(device=self$device)
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
      dataloader <- torch::dataloader(dataset, 
                                      batch_size=self$batchSize, 
                                      shuffle=TRUE)
      for (epoch in 1:self$epochs) {
        self$fitEpoch(dataloader)
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
    score = function(dataloader){
      torch::with_no_grad({
        loss = c()
        predictions = c()
        targets = c()
        self$model$eval()
        coro::loop(for (b in dataloader) {
          b <- self$batchToDevice(b)
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
      dataloader <- torch::dataloader(dataset, 
                                      batch_size = self$batchSize, 
                                      shuffle=F)
      torch::with_no_grad({
        predictions <- c()
        self$model$eval()
        coro::loop(for (b in dataloader){
          b <- self$batchToDevice(b)
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


