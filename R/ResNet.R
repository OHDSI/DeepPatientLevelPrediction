# setResNet <- function(numLayers=4, sizeHidden=64, hiddenFactor=1, 
#                       residualDropout=0.2, hiddenDropout=0.2, 
#                       normalization='BatchNorm', activation='RelU',
#                       sizeEmbedding=64, weightDecay=1e-4, 
#                       learningRate=3e-4, seed=42) {
# 
#   if (!is.null(seed)) {
#     seed <- as.integer(sample(1e5, 1))
#   }
#   
#   param <- split(expand.grid(numLayers=numLayers, sizeHidden=sizeHidden, 
#                              hiddenFactor=hiddenFactor, 
#                              residualDropout=residualDropout, 
#                              hiddenDropout=hiddenDropout, 
#                              sizeEmbedding=sizeEmbedding, wei))
#   results <- list(model='fitResNet', param=param, name='ResNet')
#   
#   class(results) <- 'modelSettings'
#   
#   return(results)
#   
# }


# fitResNet <- function(population, plpData, param, search='Random', numSearch=1,
#                       quiet=F) {
#   
#   toSparse <- toSparseM(plpData, population)
#   sparseMatrix <- toSparse$data
#   
#   outLoc <- createTempModelLoc()
#   #do cross validation to find hyperParameter
#   hyperParamSel <- lapply(param, function(x) do.call(trainResNet, 
#                                                      listAppend(x, list(plpData=sparseMatrix,
#                                                                         population = population,
#                                                                         train=TRUE,
#                                                                         modelOutput=outLoc,
#                                                                         quiet = quiet))  ))
#   hyperSummary <- cbind(do.call(rbind, param), unlist(hyperParamSel))
#   
#   
#   
#   
# }

# # trainResNet <- function(population, plpData, modelOutput, train=T) {
#   
#   
# }

ResLayer <- torch::nn_module(
  name='ResLayer',
  
  initialize=function(sizeHidden, resHidden, normalization,
                     activation, hiddenDropout=NULL, residualDropout=NULL){
    self$norm <- normalization(sizeHidden)
    self$linear0 <- torch::nn_linear(sizeHidden, resHidden)
    self$linear1 <- torch::nn_linear(resHidden, sizeHidden)
    
    self$activation <- activation
    self$hiddenDropout <- hiddenDropout
    self$residualDropout <- residualDropout
    
  },
  
  forward=function(x) {
    z <- x
    z <- self$norm(z)
    z <- self$linear0(z)
    z <- self$activation(z)
    if (!is.null(self$hiddenDropout)) {
      z <- torch::nnf_dropout(z, p=self$hiddenDropout)
    }
    z <- self$linear1(z)
    if (!is.null(self$residualDropout)) {
      z <- torch::nnf_dropout(z, p=self$residualDropout)
    }
    x <- z + x 
    return(x)
  }
)

ResNet <- torch::nn_module(
  name='ResNet',
  
  initialize=function(n_features, sizeEmbedding, sizeHidden, numLayers,
                      hiddenFactor, activation, normalization, hiddenDropout=NULL,
                      residualDropout=NULL, d_out=1) {
    # n_features - 1 because only binary features are embedded (not Age)
    # ages is concatenated with the embedding output
    # need to extend to support other numerical features
    self$embedding <- torch::nn_linear(n_features - 1, sizeEmbedding, bias=F)
    self$first_layer <- torch::nn_linear(sizeEmbedding + 1, sizeHidden)
    
    resHidden <- sizeHidden * hiddenFactor
    
    #TODO make this prettier , residualBlock class
    #TODO 
    self$layers <- torch::nn_module_list(lapply(1:numLayers,
                                                 function (x) ResLayer(sizeHidden, resHidden,
                                                          normalization, activation,
                                                          hiddenDropout,
                                                          residualDropout)))
    self$lastNorm <- normalization(sizeHidden)
    self$head <- torch::nn_linear(sizeHidden, d_out)
    
    self$lastAct <- activation

    
    
  },
      
  forward=function(x_num, x_cat) {
    x_cat <- self$embedding(x_cat)
    x <- torch::torch_cat(list(x_cat, x_num), dim=2L)
    x <- self$first_layer(x)
    
    for (i in 1:length(self$layers)) {
      x <- self$layers[[i]](x)
    }
    x <- self$lastNorm(x)
    x <- self$lastAct(x)    
    x <- self$head(x)
    x <- x$squeeze(-1)
    return(x)
  }
)

Estimator <- torch::nn_module(
  name = 'Estimator',
  initialize = function(baseModel, modelParameters, fitParameters,
                        optimizer=torch::optim_adam,
                        criterion=torch::nn_bce_with_logits_loss,
                       device='cpu'){
    self$device <- device
    self$model <- do.call(baseModel, modelParameters)
    self$modelParameters <- modelParameters
    
    self$epochs <- self$item_or_defaults(fitParameters, 'epochs', 10)
    self$learningRate <- self$item_or_defaults(fitParameters,'lr', 2e-4)
    self$l2Norm <- self$item_or_defaults(fitParameters, 'l2', 1e-5)
    
    self$resultsDir <- self$item_or_defaults(fitParameters, 'resultsDir', './results')
    dir.create(self$resultsDir)
    self$prefix <- self$item_or_defaults(fitParameters, 'prefix', 'resnet')
    
    self$previousEpochs <- self$item_or_defaults(fitParameters, 'previousEpochs', 0)
    
    self$optimizer <- optimizer(params=self$model$parameters, 
                                lr=self$learningRate, weight_decay=self$l2Norm)
    self$criterion <- criterion()
  },
  
  # fits the estimator
  fit = function(dataloader, testDataloader) {
    valLosses = c()
    valAUCs = c()
    for (epoch in 1:self$epochs) {
      self$fit_epoch(dataloader)
      scores <- self$score_epoch(testDataloader)
      
      currentEpoch <- epoch + self$previousEpochs
      
      ParallelLogger::logInfo('Epochs: ', currentEpoch, ' | Val AUC: ', 
                              round(scores$auc,3), ' | Val Loss: ', round(scores$loss,2), ' LR: ',
                              self$optimizer$param_groups[[1]]$lr)
      valLosses <- c(valLosses, scores$loss)
      valAUCs <- c(valAUCs, scores$auc)
      
      torch::torch_save(list(
        modelState_dict=self$model$state_dict(),
        modelHyperparameters=self$modelParameters),
        file.path(self$resultsDir, paste0(self$prefix, '_epochs:', currentEpoch, 
                                         '_auc:', round(scores$auc,3), '_val_loss:',
                                         round(scores$loss,2))))
                  }
    write.csv(data.frame(epochs=1:self$epochs, loss=valLosses, auc=valAUCs), 
              file.path(self$resultsDir, 'log.txt'))
  },
  
  # trains for one epoch
  fit_epoch = function(dataloader){
    t = Sys.time()
    batch_loss = 0
    i=1
    self$model$train()
    for (b in torch::enumerate(dataloader)) {
      cat = b[[1]]$to(device=self$device)
      num = b[[2]]$to(device=self$device)
      target= b[[3]]$to(device=self$device)
      
      out = self$model(num, cat)
      loss = self$criterion(out, target)
      
      batch_loss = batch_loss + loss
      if (i %% 10 == 0) {
        elapsed_time <- Sys.time() - t
        ParallelLogger::logInfo('Loss: ', round((batch_loss/10)$item(), 3), ' | Time: ',
                                round(elapsed_time,digits = 2), units(elapsed_time))
        t = Sys.time()
        batch_loss = 0
      }
      
      loss$backward()
      self$optimizer$step()
      self$optimizer$zero_grad()
      i = i + 1
    }
    
  },
  
  # calculates loss and auc after training for one epoch
  score_epoch = function(dataloader){
    torch::with_no_grad({
      loss = c()
      predictions = c()
      targets = c()
      self$model$eval()
      for (b in torch::enumerate(dataloader)) {
        cat = b[[1]]$to(device=self$device)
        num = b[[2]]$to(device=self$device)
        target = b[[3]]$to(device=self$device)
        
        pred = self$model(num, cat)
        predictions = c(predictions, as.array(pred))
        targets = c(targets, as.array(target))
        loss = c(loss, self$criterion(pred, target)$item())
      }
      mean_loss = mean(loss)
      predictionsClass = list(values=predictions, outcomeCount=targets)
      attr(predictionsClass, 'metaData')$predictionType <-'binary' 
      auc = computeAuc(predictionsClass)
    })
    return(list(loss=mean_loss, auc=auc))
  },
  
  # predicts and outputs the probabilities
  # predict_proba = function(dataloader) {
  #   
  # },
  
  # predicts and outputs the class
  # predict = function(dataloader){
  #   
  # },
  
  # select item from list, and if it's null sets a default
  item_or_defaults = function (list, item, default = NULL) {
    value = list[[item]]
    if (is.null(value)) default else value
  },
  
)

Dataset <- torch::dataset(
  name = 'Dataset',
  
  initialize=function(data, labels, indices, numericalIndex) {
    matrix <- data[indices,]
    
    tensor <- torch::torch_tensor(as.matrix(matrix), dtype=torch::torch_float32())
    
    self$labels <- torch::torch_tensor(labels)[indices]
    
    notNumIndex <- 1:tensor$shape[2] != numericalIndex
    self$cat <- tensor[, notNumIndex]
    self$num <- tensor[,numericalIndex, drop=F]
    
  },
  
  .getitem = function(item) {
    return(list(self$cat[item,], 
                self$num[item,],
                self$labels[item]))
  },
  
  .length = function() {
    self$labels$shape[1]
  }
)

  
  
  
