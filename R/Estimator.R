Estimator <- R6::R6Class('Estimator',
  public = list(
    device = NULL,
    mode = NULL,
    modelParameters = NULL,
    epochs = NULL,
    learningRate = NULL,
    l2Norm = NULL,
    batchSize = NULL,
    resultsDir = NULL,
    prefix = NULL,
    previousEpochs = NULL,
    optimizer = NULL,
    criterion = NULL,
    bestScore = NULL,
    bestEpoch = NULL,
    model = NULL,
    earlyStopper = NULL,
    initialize = function(baseModel, 
                          modelParameters, 
                          fitParameters,
                          optimizer=torch::optim_adam,
                          criterion=torch::nn_bce_with_logits_loss,
                          device='cpu', 
                          patience=3){
      self$device <- device
      self$model <- do.call(baseModel, modelParameters)
      self$modelParameters <- modelParameters
      
      self$epochs <- self$itemOrDefaults(fitParameters, 'epochs', 10)
      self$learningRate <- self$itemOrDefaults(fitParameters,'learningRate', 1e-3)
      self$l2Norm <- self$itemOrDefaults(fitParameters, 'weightDecay', 1e-5)
      self$batchSize <- self$itemOrDefaults(fitParameters, 'batchSize', 1024)
      
      # don´t save checkpoints unless you get a resultDir
      self$resultsDir <- self$itemOrDefaults(fitParameters, 'resultsDir', NULL)
      if (!is.null(self$resultsDir)) {
        dir.create(self$resultsDir, recursive=TRUE, showWarnings=FALSE) 
        }
      self$prefix <- self$itemOrDefaults(fitParameters, 'prefix', self$model$name)
      
      self$previousEpochs <- self$itemOrDefaults(fitParameters, 'previousEpochs', 0)
      
      self$optimizer <- optimizer(params=self$model$parameters, 
                                  lr=self$learningRate, 
                                  weight_decay=self$l2Norm)
      self$criterion <- criterion()
      self$earlyStopper <- EarlyStopping$new(patience=patience)
      
      self$model$to(device=self$device)
      
      self$bestScore <- NULL
      self$bestEpoch <- NULL
    },
  
    # fits the estimator
    fit = function(dataset, testDataset) {
      valLosses <- c()
      valAUCs <- c()
      
      dataloader <- torch::dataloader(dataset, 
                                      batch_size=self$batchSize, 
                                      shuffle=T)
      testDataloader <- torch::dataloader(testDataset, 
                                          batch_size=self$batchSize, 
                                          shuffle=F)
      
      modelStateDict <- list()
      epoch <- list()
      
      lr <- c()
      for (epochI in 1:self$epochs) {
        
        # fit the model
        self$fitEpoch(dataloader)
        
        # predict on test data
        scores <- self$score(testDataloader)
        
        currentEpoch <- epochI + self$previousEpochs
        lr <- c(lr, self$optimizer$param_groups[[1]]$lr)
        ParallelLogger::logInfo('Epochs: ', currentEpoch, ' | Val AUC: ', 
                                round(scores$auc,3), ' | Val Loss: ', 
                                round(scores$loss,3), ' | LR: ',
                                self$optimizer$param_groups[[1]]$lr)
        valLosses <- c(valLosses, scores$loss)
        valAUCs <- c(valAUCs, scores$auc)
        self$earlyStopper$call(scores$auc)
        if (self$earlyStopper$improved) {
          # here it saves the results to lists rather than files
          modelStateDict[[epochI]]  <- self$model$state_dict()
          epoch[[epochI]] <- currentEpoch
        }
        if (self$earlyStopper$earlyStop) {
          ParallelLogger::logInfo('Early stopping, validation AUC stopped improving')
          self$finishFit(valAUCs, modelStateDict, valLosses, epoch)
          invisible(self)
        } 
      }
      self$finishFit(valAUCs, modelStateDict, valLosses, epoch)
      invisible(self)
    },
    
    # operations that run when fitting is finished
    finishFit = function(valAUCs, modelStateDict, valLosses, epoch) {
      #extract best epoch from the saved checkpoints
      bestEpochInd <- which.max(valAUCs)  # change this if a different metric is used
      
      bestModelStateDict <- modelStateDict[[bestEpochInd]]
      self$model$load_state_dict(bestModelStateDict)
      
      bestEpoch <- epoch[[bestEpochInd]]
      self$bestEpoch <- bestEpoch
      self$bestScore <- list(loss= valLosses[bestEpochInd], auc=valAUCs[bestEpochInd])
      
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
                                      shuffle=TRUE, 
                                      drop_last=FALSE)
      for (epoch in 1:self$epochs) {
        self$fitEpoch(dataloader)
      }
      torch::torch_save(list(modelStateDict=self$model$state_dict(),
                             modelParameters=self$modelParameters,
                             fitParameters=self$fitParameters,
                             epoch=self$epochs),
                        file.path(self$resultsDir, paste0(
                          self$prefix, '_epochs:', self$epochs)
                        ))
      
    },    
    
    # trains for one epoch
    fitEpoch = function(dataloader){
      t = Sys.time()
      batch_loss = 0
      i=1
      
      self$model$train()
      
      coro::loop(for (b in dataloader) {
        cat = b[[1]]$to(device=self$device)
        num = b[[2]]$to(device=self$device)
        target = b[[3]]$to(device=self$device)
        out = self$model(num, cat)
        
        loss = self$criterion(out, target)
        
        batch_loss = batch_loss + loss
        if (i %% 10 == 0) {
          elapsed_time <- Sys.time() - t
          ParallelLogger::logInfo('Loss: ', round((batch_loss/1)$item(), 3), ' | Time: ',
                                  round(elapsed_time,digits = 2), units(elapsed_time))
          t = Sys.time()
          batch_loss = 0
        }
        
        loss$backward()
        self$optimizer$step()
        self$optimizer$zero_grad()
        i = i + 1
      })
      
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
        attr(predictionsClass, 'metaData')$predictionType <-'binary' 
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

EarlyStopping <- R6::R6Class('EarlyStopping',
   public = list(
     patience = NULL,
     delta = NULL,
     counter = NULL,
     bestScore = NULL,
     earlyStop =  NULL,
     improved = NULL,
     previousScore = NULL,
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

