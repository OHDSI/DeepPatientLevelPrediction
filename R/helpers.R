rowIdSets <- function(population, 
                     index){
  
  if(!is.null(index)){
    testRowIds <- population$rowId[population$indexes==index]
    trainRowIds <- population$rowId[population$indexes!=index]
    
    valN <- min(10000,length(trainRowIds)*0.05)
    valSamp <- sample(1:length(trainRowIds), valN, replace=FALSE)
    earlyStopRowIds <- trainRowIds[valSamp]
    trainRowIds <- trainRowIds[-valSamp]
    
    datas <- list(testRowIds = testRowIds,
                  trainRowIds = trainRowIds,
                  earlyStopRowIds = earlyStopRowIds
    )
  }else{
    trainRowIds <- population$rowId
    
    valN <- min(10000,length(trainRowIds)*0.05)
    valSamp <- sample(1:length(trainRowIds), valN, replace=FALSE)
    earlyStopRowIds <- trainRowIds[valSamp]
    trainRowIds <- trainRowIds[-valSamp]
    
    datas <- list(trainRowIds = trainRowIds,
                  earlyStopRowIds = earlyStopRowIds
    )
    
  }
  
  return(datas)
}

convertToTorchData <- function(data, label, rowIds){
  x <- torch::torch_tensor(as.matrix(data[rowIds,]), dtype = torch::torch_float())
  y <- torch::torch_tensor(label, dtype = torch::torch_float())
  return(list(x=x,
              y=y))
}

batchPredict <- function(model, 
                         plpData,
                         population,
                         predictRowIds,
                         batch_size ){
  maxVal <- length(predictRowIds)
  batches <- lapply(1:ceiling(maxVal/batch_size), function(x) ((x-1)*batch_size+1):min((x*batch_size), maxVal))
  prediction <- population[predictRowIds,]
  prediction$value <- 0
  
  for(batch in batches){
    b <- torch::torch_tensor(as.matrix(plpData[predictRowIds,][batch,,drop = F]), dtype = torch::torch_float())
    pred <- model(b)
    prediction$value[batch] <- as.array(pred$to())[,1]
  }
  attr(prediction, "metaData") <- list(predictionType = "binary")
  return(prediction)
}

updatePredictionMat <- function(predictionMat,prediction){
  predictionMat$value[prediction$rowIds] <- prediction$value
}
  
  

  
  

