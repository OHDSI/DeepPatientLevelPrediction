#' @export
setResNet <- function(numLayers=1:16, sizeHidden=2^(6:10), hiddenFactor=1:4,
                      residualDropout=seq(0,0.3,0.05), hiddenDropout=seq(0,0.3,0.05),
                      normalization='BatchNorm', activation='RelU',
                      sizeEmbedding=2^(6:9), weightDecay=c(1e-6, 1e-3),
                      learningRate=c(1e-2,1e-5), seed=42, hyperParamSearch='random',
                      randomSample=100, device='cpu', batch_size=1024, epochs=10) {

  if (!is.null(seed)) {
    seed <- as.integer(sample(1e5, 1))
  }

  param <- expand.grid(numLayers=numLayers, sizeHidden=sizeHidden,
                             hiddenFactor=hiddenFactor,
                             residualDropout=residualDropout,
                             hiddenDropout=hiddenDropout,
                             sizeEmbedding=sizeEmbedding, weightDecay=weightDecay,
                             learningRate=learningRate)
  if (hyperParamSearch=='random'){
    param <- param[sample(nrow(param), randomSample),]
  }
  param$device <- device
  param$batch_size <- batch_size
  param$epochs <- epochs
  
  results <- list(model='fitResNet', param=param, name='ResNet')

  class(results) <- 'modelSettings'

  return(results)

}

#' @export
fitResNet <- function(population, plpData, param,
                      quiet=F, outcomeId, cohortId, ...) {
  
  start <- Sys.time()
  sparseMatrix <- toSparseM(plpData, population)

  
  # TODO where to save results?
  outLoc <- './results'
  
  #do cross validation to find hyperParameters
  hyperParamSel <- list()
  for (i in 1:nrow(param)) {
    outLocHP <- file.path(outLoc, paste0('Iteration_', i))
    hyperParamSel[[i]] <- do.call(trainResNet, listAppend(param[i,], list(sparseMatrix =sparseMatrix,
                                                                           population = population,
                                                                           train=TRUE,
                                                                           modelOutput=outLocHP,
                                                                           quiet = quiet)))
  }
  hyperSummary <-as.data.frame(cbind(do.call(rbind, lapply(hyperParamSel, function(x) x$hyperSum))))
  hyperSummary$auc <- unlist(lapply(hyperParamSel, function(x) x$auc))
  
  scores <- unlist(lapply(hyperParamSel, function(x) x$auc))
  
  # now train the final model and return coef
  bestInd <- which.max(abs(unlist(scores)-0.5))[1]
  uniqueEpochs <- unique(hyperSummary$bestEpochs[[bestInd]])
  param$epochs <- uniqueEpochs[which.max(tabulate(match(hyperSummary$bestEpochs[[bestInd]], uniqueEpochs)))]
  outLoc <- file.path(outLoc, paste0('whole_training_set'))
  finalModel <- do.call(trainResNet, listAppend(param[bestInd,], 
                                                  list(sparseMatrix = sparseMatrix,
                                                       population = population,
                                                       train=FALSE,
                                                       modelOutput=outLoc)))
  covariateRef <- as.data.frame(plpData$covariateData$covariateRef)
  incs <- rep(1, nrow(covariateRef)) 
  covariateRef$included <- incs
  covariateRef$covariateValue <- rep(0, nrow(covariateRef))
  
  modelTrained <- file.path(outLoc) 
  param.best <- param[bestInd,]
  
  comp <- Sys.time() - start
  # return model location
  result <- list(model = finalModel$model,
                 trainCVAuc = scores[bestInd],
                 hyperParamSearch = hyperSummary,
                 modelSettings = list(model='fitResNet',modelParameters=param.best),
                 metaData = plpData$metaData,
                 populationSettings = attr(population, 'metaData'),
                 outcomeId=outcomeId,
                 cohortId=cohortId,
                 varImp = covariateRef, 
                 trainingTime =comp,
                 covariateMap=sparseMatrix$map, # I think this is need for new data to map the same?
                 predictionTrain = finalModel$prediction
  )
  class(result) <- 'plpModel'
  attr(result, 'type') <- 'deepEstimator'
  attr(result, 'predictionType') <- 'binary'
  return(result)
}

#' @export
trainResNet <- function(sparseMatrix, population,...,train=T) {

  param <- list(...)

  modelParamNames <- c("numLayers", "sizeHidden", "hiddenFactor",
                      "residualDropout", "hiddenDropout", "sizeEmbedding")
  
  # TODO can I use lapply here instead of for loops?
  modelParam <- list()
  for (i in 1:length(modelParamNames)){
    modelParam[i] <- param[modelParamNames[i]]
  }
  names(modelParam) <- modelParamNames

  fitParamNames <- c("weightDecay", "learningRate", "epochs")
  fitParams <- list()
  for (i in 1:length(fitParamNames)) {
    fitParams[i] <- param[fitParamNames[i]]
  }
  names(fitParams) <- fitParamNames


  n_features <- ncol(sparseMatrix$data)
  modelParam$n_features <- n_features
  
  # TODO make more general for other variables than only age
  numericalIndex <- sparseMatrix$map$newCovariateId[sparseMatrix$map$oldCovariateId==1002]
  
  index_vect <- unique(population$indexes[population$indexes > 0])
  if(train==T){
    ParallelLogger::logInfo(paste('Training deep neural network using Torch with ',length(index_vect),' fold CV'))
    foldAuc <- c()
    foldEpochs <- c()
    for(index in 1:length(index_vect)){
      fitParams$resultsDir <- file.path(param$modelOutput, paste0('fold_', index))
      ParallelLogger::logInfo(paste('Fold ',index, ' -- with ', sum(population$indexes!=index),'train rows'))
      estimator <- Estimator(baseModel=ResNet, modelParameters=modelParam,
                         fitParameters=fitParams, device=param$device)
      testIndices <- population$rowId[population$indexes==index]
      trainIndices <- population$rowId[(population$indexes!=index) & (population$indexes > 0)]
      trainDataset <- Dataset(sparseMatrix$data, population$outcomeCount, indices=trainIndices, numericalIndex=numericalIndex)
      testDataset <- Dataset(sparseMatrix$data, population$outcomeCount, indices=testIndices, numericalIndex=numericalIndex)
      trainDataloader <- torch::dataloader(trainDataset, batch_size=param$batch_size, shuffle=T, drop_last=TRUE)
      testDataloader <- torch::dataloader(testDataset, batch_size=param$batch_size, shuffle=F)
      
      score <- estimator$fit(trainDataloader, testDataloader)$score(testDataloader)
      bestEpoch <- estimator$bestEpoch
      auc <- score$auc
      foldAuc <- c(foldAuc, auc)
      foldEpochs <- c(foldEpochs, bestEpoch)
    }
    auc <- mean(foldAuc)
    predictions <- NULL
    bestEpochs <- list(bestEpochs=foldEpochs)
  }
  else {
    ParallelLogger::logInfo('Training deep neural network using Torch on whole training set')
    fitParams$resultsDir <- param$modelOutput      
    estimator <- Estimator(baseModel = ResNet, modelParameters = modelParam,
                           fitParameters = fitParams, device=param$device)
    trainIndices <- population$rowId[population$indexes > 0]
    
    trainDataset <- Dataset(sparseMatrix$data, population$outcomeCount, indices=trainIndices, numericalIndex=numericalIndex)
    trainDataloader <- torch::dataloader(trainDataset, batch_size=param$batch_size, shuffle=T, drop_last=TRUE)

    estimator$fitWholeTrainingSet(trainDataloader, param$epochs)
    dataloader <- torch::dataloader(trainDataset, batch_size = param$batch_size, shuffle=F, drop_last=FALSE)
    predictions <- population[trainIndices, ]
    predictions$value <- estimator$predictProba(dataloader)
    predictionsClass <- list(value=predictions$value, outcomeCount=as.array(trainDataset$labels))
    attr(predictionsClass, 'metaData')$predictionType <-'binary' 
    auc <- computeAuc(predictionsClass)
    bestEpochs <- NULL
  }
 
   result <- list(model=estimator,
                 auc = auc,
                 prediction = predictions,
                 hyperSum = c(modelParam, fitParams, bestEpochs))

  return(result)
  }

ResLayer <- torch::nn_module(
  name='ResLayer',
  
  initialize=function(sizeHidden, resHidden, normalization,
                     activation, hiddenDropout=NULL, residualDropout=NULL){
    self$norm <- normalization(sizeHidden)
    self$linear0 <- torch::nn_linear(sizeHidden, resHidden)
    self$linear1 <- torch::nn_linear(resHidden, sizeHidden)
    
    self$activation <- activation
    if (!is.null(hiddenDropout)){
      self$hiddenDropout <- torch::nn_dropout(p=hiddenDropout)
    }
    if (!is.null(residualDropout)) 
    {
      self$residualDropout <- torch::nn_dropout(p=residualDropout)
    }
    
    self$activation <- activation()
    
    
    
  },
  
  forward=function(x) {
    z <- x
    z <- self$norm(z)
    z <- self$linear0(z)
    z <- self$activation(z)
    if (!is.null(self$hiddenDropout)) {
      z <- self$hiddenDropout(z)
    }
    z <- self$linear1(z)
    if (!is.null(self$residualDropout)) {
      z <- self$residualDropout(z)
    }
    x <- z + x 
    return(x)
  }
)

ResNet <- torch::nn_module(
  name='ResNet',
  
  initialize=function(n_features, sizeEmbedding, sizeHidden, numLayers,
                      hiddenFactor, activation=torch::nn_relu, 
                      normalization=torch::nn_batch_norm1d, hiddenDropout=NULL,
                      residualDropout=NULL, d_out=1) {
    # n_features - 1 because only binary features are embedded (not Age)
    # ages is concatenated with the embedding output
    # TODO need to extend to support other numerical features
    self$embedding <- torch::nn_linear(n_features - 1, sizeEmbedding, bias=F)
    self$first_layer <- torch::nn_linear(sizeEmbedding + 1, sizeHidden)
    
    resHidden <- sizeHidden * hiddenFactor
    
    self$layers <- torch::nn_module_list(lapply(1:numLayers,
                                                 function (x) ResLayer(sizeHidden, resHidden,
                                                          normalization, activation,
                                                          hiddenDropout,
                                                          residualDropout)))
    self$lastNorm <- normalization(sizeHidden)
    self$head <- torch::nn_linear(sizeHidden, d_out)
    
    self$lastAct <- activation()
    
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
    
    self$epochs <- self$itemOrDefaults(fitParameters, 'epochs', 10)
    self$learningRate <- self$itemOrDefaults(fitParameters,'learningRate', 1e-3)
    self$l2Norm <- self$itemOrDefaults(fitParameters, 'weightDecay', 1e-5)
    
    self$resultsDir <- self$itemOrDefaults(fitParameters, 'resultsDir', './results')
    dir.create(self$resultsDir, recursive=TRUE, showWarnings=FALSE)
    self$prefix <- self$itemOrDefaults(fitParameters, 'prefix', 'resnet')
    
    self$previousEpochs <- self$itemOrDefaults(fitParameters, 'previousEpochs', 0)
    
    self$optimizer <- optimizer(params=self$model$parameters, 
                                lr=self$learningRate, weight_decay=self$l2Norm)
    self$criterion <- criterion()
    self$model$to(device=self$device)
  },
  
  # fits the estimator
  fit = function(dataloader, testDataloader) {
    valLosses <- c()
    valAUCs <- c()
    lr <- c()
    for (epoch in 1:self$epochs) {
      self$fitEpoch(dataloader)
      scores <- self$score(testDataloader)
      
      currentEpoch <- epoch + self$previousEpochs
      lr <- c(lr, self$optimizer$param_groups[[1]]$lr)
      ParallelLogger::logInfo('Epochs: ', currentEpoch, ' | Val AUC: ', 
                              round(scores$auc,3), ' | Val Loss: ', 
                              round(scores$loss,3), ' | LR: ',
                              self$optimizer$param_groups[[1]]$lr)
      valLosses <- c(valLosses, scores$loss)
      valAUCs <- c(valAUCs, scores$auc)
      
      torch::torch_save(list(
        modelStateDict=self$model$state_dict(),
        modelHyperparameters=self$modelParameters,
        epoch=currentEpoch),
        file.path(self$resultsDir, paste0(self$prefix, '_epochs:', currentEpoch, 
                                         '_auc:', round(scores$auc,4), '_val_loss:',
                                         round(scores$loss,4))))
                  }
    write.csv(data.frame(epochs=1:self$epochs, loss=valLosses, auc=valAUCs), 
              file.path(self$resultsDir, 'log.txt'))
    
    #TODO here I should extract best epoch from the saved checkpoints
    bestModelFile <- self$extractBestModel(metric='val_loss')
    bestModel <- torch::torch_load(bestModelFile)
    bestModelStateDict <- bestModel$modelStateDict
    self$model$load_state_dict(bestModelStateDict)
    bestEpoch <- bestModel$epoch
    ParallelLogger::logInfo(paste0('Loaded best model from epoch ', bestEpoch))
    self$bestEpoch <- bestEpoch
    
    invisible(self)
  },
  
  # Fits whole training set on a specific number of epochs
  # TODO What happens when learning rate changes per epochs?
  # Ideally I would copy the learning rate strategy from before
  fitWholeTrainingSet = function(dataloader, epochs) {
    for (epoch in 1:epochs) {
      self$fitEpoch(dataloader)
    }
    
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
      target= b[[3]]$to(device=self$device)
      out = self$model(num, cat)
      loss = self$criterion(out, target)
      
       batch_loss = batch_loss + loss
      if (i %% 1 == 10) {
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
      predictionsClass <- list(value=predictions, outcomeCount=targets)
      attr(predictionsClass, 'metaData')$predictionType <-'binary' 
      auc <- computeAuc(predictionsClass)
    })
    return(list(loss=mean_loss, auc=auc))
  },
  
  # predicts and outputs the probabilities
  predictProba = function(dataloader) {
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
  predict = function(dataloader){
    predictions <- self$predict_proba(dataloader)
    predicted_class <- torch::torch_argmax(torch::torch_unsqueeze(torch::torch_tensor(predictions), dim=2),dim=2)
    return(predicted_class)
  },
  
  load_best_weight = function(){
    best_model_file <- self$extract_best_model(self$resultsDor)
    best_model <- torch::torch_load(best_model_file)
    state_dict <- best_model$model_state_dict
    epoch <- best_model$epoch
    self$model$load_state_dict(state_dict)
    ParallelLogger::logInfo(paste('Loaded best model from epoch: ', epoch))
  },
  
  # extracts best model from the results directory
  extractBestModel = function(metric='val_loss'){

    if (metric=='val_loss')
    {
      direction = 'min'
    }
    else
    {
      direction = 'max'
    }

    # goes over checkpoints in folder and extracts metric value from name
    checkpoints <- Sys.glob(file.path(self$resultsDir, paste0('*', metric, '*')))
    metric_value <- c()
    for (file in checkpoints) {
      fileName <- basename(file)
      metric_value <- c(metric_value, as.double(strsplit(strsplit(fileName, paste0(metric, ':'))[[1]][2], '_')[[1]][1]))
    }
    
    if (direction == 'max') {
      best_index <- which.max(metric_value)
    } else if (direction == 'min') {
      best_index <- which.min(metric_value)
    }
    bestModel <- checkpoints[[best_index]]
    
    return(bestModel)
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
    value = list[[item]]
    if (is.null(value)) default else value
  },
  
)

Dataset <- torch::dataset(
  name = 'Dataset',
  
  initialize=function(data, labels, indices, numericalIndex) {
    matrix <- data[indices,]
    
    tensor <- torch::torch_tensor(as.matrix(matrix), dtype=torch::torch_float32())
    
    # if labels have already been restricted to population
    if (max(indices)>length(labels)) {
     self$labels <- torch::torch_tensor(labels) 
    }
    else {
      self$labels <- torch::torch_tensor(labels)[indices]
    }
    
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

  
  
  
