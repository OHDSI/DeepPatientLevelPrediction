setResNet <- function(numLayers=1:16, sizeHidden=2^(6:10), hiddenFactor=1:4,
                      residualDropout=seq(0,0.3,0.05), hiddenDropout=seq(0,0.3,0.05),
                      normalization='BatchNorm', activation='RelU',
                      sizeEmbedding=2^(6:9), weightDecay=c(1e-6, 1e-3),
                      learningRate=c(1e-2,1e-5), seed=42, hyperParamSearch='random',
                      randomSample=100, device='cpu', batch_size=1024) {

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

  results <- list(model='fitResNet', param=param, name='ResNet')

  class(results) <- 'modelSettings'

  return(results)

}


fitResNet <- function(population, plpData, param,
                      quiet=F) {

  toSparse <- toSparseM(plpData, population)
  sparseMatrix <- toSparse$data

  outLoc <- createTempModelLoc()
  #do cross validation to find hyperParameter
  hyperParamSel <- lapply(param, function(x) do.call(trainResNet,
                                                     listAppend(x, list(plpData=sparseMatrix,
                                                                        population = population,
                                                                        train=TRUE,
                                                                        modelOutput=outLoc,
                                                                        quiet = quiet))  ))
  hyperSummary <- cbind(do.call(rbind, param), unlist(hyperParamSel))

  #now train the final model and return coef
  bestInd <- which.max(abs(unlist(hyperParamSel)-0.5))[1]
  finalModel <- do.call(trainCNNTorch, listAppend(param[[bestInd]], 
                                                  list(plpData = result$data,
                                                       population = Population,
                                                       train=FALSE,
                                                       modelOutput=outLoc)))
  covariateRef <- as.data.frame(plpData$covariateData$covariateRef)
  incs <- rep(1, nrow(covariateRef)) 
  covariateRef$included <- incs
  covariateRef$covariateValue <- rep(0, nrow(covariateRef))
  
  modelTrained <- file.path(outLoc) 
  param.best <- param[[bestInd]]
  
  comp <- start-Sys.time()
  
  # train prediction
  pred <- as.matrix(finalModel)
  pred[,1] <- pred[,1]
  colnames(pred) <- c('rowId','outcomeCount','indexes', 'value')
  pred <- as.data.frame(pred)
  attr(pred, "metaData") <- list(predictionType="binary")
  
  pred$value <- 1-pred$value
  prediction <- merge(population, pred[,c('rowId','value')], by='rowId')
  
  # return model location 
  result <- list(model = modelTrained,
                 trainCVAuc = -1, # ToDo decide on how to deal with this
                 hyperParamSearch = hyperSummary,
                 modelSettings = list(model='fitResNet',modelParameters=param.best),
                 metaData = plpData$metaData,
                 populationSettings = attr(population, 'metaData'),
                 outcomeId=outcomeId,
                 cohortId=cohortId,
                 varImp = covariateRef, 
                 trainingTime =comp,
                 dense=1,
                 covariateMap=result$map, # I think this is need for new data to map the same?
                 predictionTrain = prediction
  )
  class(result) <- 'plpModel'
  attr(result, 'predictionType') <- 'binary'

  return(result)
}

trainResNet <- function(population, plpData, modelOutput, train=T, ...) {

  param <- list(...)

  modelParamNames <- c("numLayers", "sizeHidden", "hiddenFactor",
                      "residualDropout", "hiddenDropout", "sizeEmbedding")
  # TODO can I use lapply here instead of for loops?
  modelParam <- list()
  for (i in 1:length(modelParamNames)){
    modelParam[[i]] <- param[,modelParamNames[[i]]]
  }
  names(modelParam) <- modelParamNames

  fitParamNames <- c("weightDecay", "learningRate")
  fitParams <- list()
  for (i in 1:length(fitParamNames)) {
    fitParams[[i]] <- param[, fitParamNames[[i]]]
  }
  names(fitParams) <- fitParamNames

  fitParams$resultDir <- modelOutput

  sparseM <- toSparseM(plpData, population, temporal=F)
  n_features <- nrow(sparseM$data)
  modelParams$n_features <- n_features
  
  # TODO make more general for other variables than only age
  numericalIndex <- sparseM$map$newCovariateId[sparseM$map$oldCovariateId==1002]
  
  if(!is.null(population$indexes) && train==T){
    index_vect <- unique(population$index)
    ParallelLogger::logInfo(paste('Training deep neural network using Torch with ',length(index_vect[index_vect>0]),' fold CV'))
    
    foldAuc <- c()
    for(index in 1:length(index_vect)){
      ParallelLogger::logInfo(paste('Fold ',index, ' -- with ', sum(population$indexes!=index),'train rows'))
      estimator <- Estimator(baseModel=ResNet, modelParameters=modelParam,
                         fitParameters=fitParams, device=param$device)
      testIndices <- population$rowId[population$index==index]
      trainIndices <- population$rowId[(population$index!=index) & (population$index > 0)]
      trainDataset <- Dataset(sparseM$data, population$outcomeCount, indices=trainIndices, numericalIndex=numericalIndex)
      testDataset <- Dataset(sparseM$data, population$outcomeCount, indices=trainIndices, numericalIndex=numericalIndex)
      trainDataloader <- torch::dataloader(trainDataset, batch_size=param$batch_size, shuffle=T, drop_last=TRUE)
      testDataloader <- torch::dataloader(testDataset, batch_size=param$batch_size, shuffle=F)
      
      estimator.fit(trainDataloader, testDataloader)
      score <- estimator.score(testDataloader)
      
      auc <- score$auc
      foldAuc <- c(foldAuc, auc)
    }
  }
  
  result <- list(model=estimator,
                 auc = mean(foldauc),
                 prediction = NULL,
                 hyperSum = c(modelParam, fitParams))
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
    
    self$epochs <- self$item_or_defaults(fitParameters, 'epochs', 10)
    self$learningRate <- self$item_or_defaults(fitParameters,'learningRate', 1e-3)
    self$l2Norm <- self$item_or_defaults(fitParameters, 'weightDecay', 1e-5)
    
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
  score_epoch = function(dataloader){
    torch::with_no_grad({
      loss = c()
      predictions = c()
      targets = c()
      self$model$eval()
      coro::loop(for (b in dataloader) {
        cat = b[[1]]$to(device=self$device)
        num = b[[2]]$to(device=self$device)
        target = b[[3]]$to(device=self$device)
        
        pred = self$model(num, cat)
        predictions = c(predictions, as.array(pred$cpu()))
        targets = c(targets, as.array(target$cpu()))
        loss = c(loss, self$criterion(pred, target)$item())
      })
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

  
  
  
