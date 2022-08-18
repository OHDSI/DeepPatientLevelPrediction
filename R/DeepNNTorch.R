#' settings for a Deep neural network
#' @param units           A list of vectors for neurons per layer
#' @param layerDropout   Dropout to use per layer
#' @param lr              Learning rate ot use
#' @param decay           Weight decay to use
#' @param outcomeWeight  Weight for minority outcome in cost function
#' @param batchSize      Batch size to use
#' @param epochs          How many epochs to use
#' @param device          Which device to use
#' @param seed            A seed to make experiments more reproducible
#' @export
setDeepNNTorch <- function(units = list(c(128, 64), 128),
                           layerDropout = c(0.2),
                           lr = c(1e-4),
                           decay = c(1e-5),
                           outcomeWeight = c(1.0),
                           batchSize = c(10000),
                           epochs = c(100),
                           device = "cpu",
                           seed = NULL) {
  param <- expand.grid(
    units = units,
    layerDropout = layerDropout,
    lr = lr, decay = decay, outcomeWeight = outcomeWeight, epochs = epochs,
    seed = ifelse(is.null(seed), "NULL", seed)
  )

  param$units1 <- unlist(lapply(param$units, function(x) x[1]))
  param$units2 <- unlist(lapply(param$units, function(x) x[2]))
  param$units3 <- unlist(lapply(param$units, function(x) x[3]))
  param$units <- NULL

  attr(param, "settings") <- list(
    modelType = "DeepNN",
    seed = seed[1],
    name = "DeepNNTorch",
    units = units,
    layerDropout = layerDropout,
    lr = lr,
    decay = decay,
    outcomeWeight = outcomeWeight,
    batchSize = batchSize,
    device = device,
    epochs = epochs
  )

  attr(param, "modelType") <- "binary"
  attr(param, "settings")$saveType <- "file"

  result <- list(
    fitFunction = "fitDeepNNTorch",
    param = param
  )

  class(result) <- "modelSettings"

  return(result)
}

#' Fits a deep neural network
#' @param trainData     Training data object
#' @param modelSettings modelSettings object
#' @param search        Which kind of search strategy to use
#' @param analysisId    Analysis Id
#' @export
fitDeepNNTorch <- function(trainData,
                           modelSettings,
                           search = "grid",
                           analysisId) {
  start <- Sys.time()

  # check covariateData
  if (!FeatureExtraction::isCovariateData(trainData$covariateData)) {
    stop("DeepNNTorch requires correct covariateData")
  }

  param <- modelSettings$param
  # get the settings from the param
  settings <- attr(param, "settings")

  if (!is.null(trainData$folds)) {
    trainData$labels <- merge(trainData$labels, trainData$fold, by = "rowId")
  }

  mappedData <- PatientLevelPrediction::toSparseM(
    plpData = trainData,
    map = NULL
  )

  matrixData <- mappedData$dataMatrix
  labels <- mappedData$labels
  covariateRef <- mappedData$covariateRef

  outLoc <- PatientLevelPrediction::createTempModelLoc()

  cvResult <- do.call(
    what = gridCvDeepNN,
    args = list(
      matrixData = matrixData,
      labels = labels,
      seed = settings$seed,
      modelName = settings$name,
      device = settings$device,
      batchSize = settings$batchSize,
      epochs = settings$epochs,
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
    model = cvResult$estimator, # file.path(outLoc),

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
        extraSettings = attr(param, "settings")
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
  attr(result, "predictionFunction") <- "predictDeepNN"
  attr(result, "modelType") <- "binary"
  attr(result, "saveType") <- attr(param, "settings")$saveType

  return(result)
}

#' Create predictions for a deep neural network
#' @param plpModel  The plpModel to predict for
#' @param data      The data to make predictions for
#' @param cohort    The cohort to use
#' @export
predictDeepNN <- function(plpModel,
                          data,
                          cohort,
                          batchSize=512,
                          device='cpu') {
  if (!inherits(plpModel, 'plpModel') & !inherits(plpModel, 'nn_module')) {
    plpModel <- list(model = plpModel)
    attr(plpModel, "modelType") <- "binary"
  }

  if (inherits(data, 'plpData')) {
    dataMat <- PatientLevelPrediction::toSparseM(
      plpData = data,
      cohort = cohort,
      map = plpModel$covariateImportance %>%
        dplyr::select(.data$columnId, .data$covariateId)
    )

    data <- Dataset(dataMat$dataMatrix, all = TRUE) # add numeric details..
  }

  # get predictions
  prediction <- cohort

  if (is.character(plpModel$model)) {
    model <- torch::torch_load(file.path(plpModel$model, "DeepNNTorchModel.pt"), device = "cpu")
  } else {
    model <- plpModel
  }
  model$to(device=device)
  batchIndex <- 1:length(data)
  batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex) / batchSize))
  torch::with_no_grad({
    predictions <- c()
    model$eval()
    coro::loop(for (b in batchIndex) {
      batch <- data[b]$batch$all$to(device=device)
      target <- data[b]$target$to(device=device)
      pred <- model(batch)
      predictions <- c(predictions, as.array(torch::torch_sigmoid(pred[,1]$cpu())))
    })
  })
  prediction$value <- predictions

  attr(prediction, "metaData")$modelType <- attr(plpModel, "modelType")

  return(prediction)
}


gridCvDeepNN <- function(matrixData,
                         labels,
                         seed,
                         modelName,
                         device,
                         batchSize,
                         epochs,
                         modelLocation,
                         paramSearch) {
  ParallelLogger::logInfo(paste0("Running CV for ", modelName, " model"))

  ###########################################################################


  gridSearchPredictons <- list()
  length(gridSearchPredictons) <- nrow(paramSearch)

  for (gridId in 1:nrow(paramSearch)) {

    # get the params
    modelParamNames <- c("layerDropout", "lr", "decay", "outcomeWeight", "epochs", "units1", "units2", "units3")
    modelParams <- paramSearch[gridId, modelParamNames]

    fitParams <- paramSearch[gridId, c("lr", "decay")]
    fitParams$epochs <- epochs
    fitParams$batchSize <- batchSize


    # initiate prediction
    prediction <- c()

    fold <- labels$index
    ParallelLogger::logInfo(paste0("Max fold: ", max(fold)))

    dataset <- Dataset(matrixData, labels$outcomeCount, all = TRUE)
    # modelParams$cat_features <- dataset$cat$shape[2]
    # modelParams$num_features <- dataset$num$shape[2]

    for (i in 1:max(fold)) {
      if (is.na(modelParams$units2)) {
        model <- singleLayerNN(
          inputN = ncol(matrixData),
          layer1 = modelParams$units1,
          outputN = 2,
          layer_dropout = modelParams$layerDropout
        )
      } else if (is.na(modelParams$units3)) {
        model <- doubleLayerNN(
          inputN = ncol(matrixData),
          layer1 = modelParams$units1,
          layer2 = modelParams$units2,
          outputN = 2,
          layer_dropout = modelParams$layerDropout
        )
      } else {
        model <- tripleLayerNN(
          inputN = ncol(matrixData),
          layer1 = modelParams$units1,
          layer2 = modelParams$units2,
          layer3 = modelParams$units3,
          outputN = 2,
          layer_dropout = modelParams$layerDropout
        )
      }
      
      model$to(device=device)
      criterion <- torch::nn_bce_loss() # Binary crossentropy only
      optimizer <- torch::optim_adam(model$parameters, lr = fitParams$lr,
                                     weight_decay = fitParams$decay)

      # Need earlyStopping
      # Need setting decay

      ParallelLogger::logInfo(paste0("Fold ", i))
      trainDataset <- torch::dataset_subset(dataset, indices = which(fold != i))
      testDataset <- torch::dataset_subset(dataset, indices = which(fold == i))

      batchIndex <- torch::torch_randperm(length(trainDataset)) + 1L
      batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex) / batchSize))
      
      testBatchIndex <- 1:length(testDataset)
      testBatchIndex <- split(testBatchIndex, ceiling(seq_along(testBatchIndex) / batchSize))
      for (j in 1:epochs) {
        startTime <- Sys.time()
        trainLosses <- torch::torch_empty(length(batchIndex))
        ix <- 1
        model$train()
        progressBar <- utils::txtProgressBar(style = 3)
        coro::loop(for (b in batchIndex) {
          optimizer$zero_grad()
          batch <- trainDataset[b]$batch$all$to(device=device)
          target <- trainDataset[b]$target$to(device=device)
          y_pred <- model(batch)
          loss <- criterion(y_pred[, 1], target)
          loss$backward()
          optimizer$step()
          
          trainLosses[ix] <- loss$detach()
          utils::setTxtProgressBar(progressBar, ix / length(batchIndex))
          ix <- ix + 1
        })
        close(progressBar)
        trainLoss <- trainLosses$mean()$item()
        torch::with_no_grad({
          ix <- 1
          testLosses <- torch::torch_empty(length(batchIndex))
          model$eval()
          predictions <- list()
          targets <- list()
          coro::loop(for (b in testBatchIndex) {
            batch <- dataset[b]$batch$all$to(device=device)
            target <- dataset[b]$target$to(device=device)
            pred <- model(batch)
            predictions <- c(predictions, pred[,1])
            targets <- c(targets, target)
            testLosses[ix] <- criterion(pred[,1], target)
            ix <- ix + 1
          })
          testLoss <- loss$mean()$item()
          predictionsClass <- data.frame(
            value = as.matrix(torch::torch_sigmoid(torch::torch_cat(predictions)$cpu())),
            outcomeCount = as.matrix(torch::torch_cat(targets)$cpu())
          )
          attr(predictionsClass, "metaData")$modelType <- "binary"
          auc <- PatientLevelPrediction::computeAuc(predictionsClass)
        })
        
        delta <- Sys.time() - startTime
        ParallelLogger::logInfo(
          "Epochs: ", j,
          " | Val AUC: ", round(auc, 3),
          " | Val Loss: ", round(testLoss, 3),
          " | Train Loss: ", round(trainLoss, 3),
          " | Time: ", round(delta, 3), " ",
          units(delta)
        )
        
      }
      
      predictionTable <- labels[labels$index == i, ]
      predictionTable$value <- predictionsClass$value

      if (!"plpModel" %in% class(model)) {
        model <- list(model = model)
        attr(model, "modelType") <- "binary"
      }
      attr(predictionTable, "metaData")$modelType <- attr(model, "modelType")

      prediction <- rbind(prediction, predictionTable)
    }
    gridSearchPredictons[[gridId]] <- list(
      prediction = prediction,
      param = paramSearch[gridId, ]
    )
  }


  # get best para (this could be modified to enable any metric instead of AUC, just need metric input in function)
  paramGridSearch <- lapply(gridSearchPredictons, function(x) {
    do.call(PatientLevelPrediction::computeGridPerformance, x)
  }) # cvAUCmean, cvAUC, param

  optimalParamInd <- which.max(unlist(lapply(paramGridSearch, function(x) x$cvPerformance)))

  finalParam <- paramGridSearch[[optimalParamInd]]$param

  cvPrediction <- gridSearchPredictons[[optimalParamInd]]$prediction
  cvPrediction$evaluationType <- "CV"

  ParallelLogger::logInfo("Training final model using optimal parameters")
  
  # get the params
  modelParams <- finalParam[modelParamNames]
  fitParams <- finalParam[c("lr", "decay")]
  fitParams$epochs <- epochs
  fitParams$batchSize <- batchSize
  # create the dir
  if (!dir.exists(file.path(modelLocation))) {
    dir.create(file.path(modelLocation), recursive = T)
  }

  trainDataset <- Dataset(
    matrixData,
    labels$outcomeCount,
    all = TRUE
  )

  # modelParams$cat_features <- trainDataset$cat$shape[2]
  # modelParams$num_features <- trainDataset$num$shape[2]

  # trainDataset <- torch::dataset_subset(dataset, indices=which(fold!=i))

  if (is.na(modelParams$units2)) {
    model <- singleLayerNN(
      inputN = ncol(matrixData),
      layer1 = modelParams$units1,
      outputN = 2,
      layer_dropout = modelParams$layerDropout
    )
  } else if (is.na(modelParams$units3)) {
    model <- doubleLayerNN(
      inputN = ncol(matrixData),
      layer1 = modelParams$units1,
      layer2 = modelParams$units2,
      outputN = 2,
      layer_dropout = modelParams$layerDropout
    )
  } else {
    model <- tripleLayerNN(
      inputN = ncol(matrixData),
      layer1 = modelParams$units1,
      layer2 = modelParams$units2,
      layer3 = modelParams$units3,
      outputN = 2,
      layer_dropout = modelParams$layerDropout
    )
  }

  model$to(device=device)
  
  criterion <- torch::nn_bce_loss() # Binary crossentropy only
  optimizer <- torch::optim_adam(model$parameters, lr = fitParams$lr)
  
  batchIndex <- torch::torch_randperm(length(trainDataset)) + 1L
  batchIndex <- split(batchIndex, ceiling(seq_along(batchIndex) / batchSize))
  
  for (epoch in 1:epochs) {
    ix <- 1
    model$train()
    progressBar <- utils::txtProgressBar(style = 3)
    coro::loop(for (b in batchIndex) {
      optimizer$zero_grad()
      batch <- dataset[b]$batch$all$to(device=device)
      target <- dataset[b]$target$to(device=device)
      out <- model(batch)
      loss <- criterion(out[,1], target)
      loss$backward()
      
      optimizer$step()
      utils::setTxtProgressBar(progressBar, ix / length(batchIndex))
      ix <- ix + 1
    })
    close(progressBar)
  }
  
  browser()
  ParallelLogger::logInfo("Calculating predictions on all train data...")

  prediction <- predictDeepNN(model, data=trainDataset, cohort=labels, 
                              batchSize = batchSize, device = device)
  prediction$evaluationType <- "Train"

  prediction <- rbind(
    prediction,
    cvPrediction
  )

  # modify prediction
  prediction <- prediction %>%
    dplyr::select(-.data$rowId, -.data$index) %>%
    dplyr::rename(rowId = .data$originalRowId)

  prediction$cohortStartDate <- as.Date(prediction$cohortStartDate, origin = "1970-01-01")


  # save torch code here
  torch::torch_save(model, file.path(modelLocation, "DeepNNTorchModel.pt"))

  return(
    list(
      estimator = modelLocation,
      prediction = prediction,
      finalParam = finalParam,
      paramGridSearch = paramGridSearch
    )
  )
}
