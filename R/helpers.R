rowIdSets <- function(population, 
                     index){
  
  if(!is.null(index)){
    testRowIds <- population$rowId[population$indexes==index]
    trainRowIds <- population$rowId[population$indexes!=index]
    
    valN <- min(10000,length(trainRowIds)*0.05)
    valSamp <- sample(1:length(trainRowIds), valN, replace=FALSE)
    earlyStopRowIds <- trainRowIds[valSamp]
    trainRowIds <- trainRowIds[-valSamp]
    
    datas <- list(testRowIds = sort(testRowIds),
                  trainRowIds = sort(trainRowIds),
                  earlyStopRowIds = sort(earlyStopRowIds)
    )
  }else{
    trainRowIds <- population$rowId
    
    valN <- min(10000,length(trainRowIds)*0.05)
    valSamp <- sample(1:length(trainRowIds), valN, replace=FALSE)
    earlyStopRowIds <- trainRowIds[valSamp]
    trainRowIds <- trainRowIds[-valSamp]
    
    datas <- list(trainRowIds = sort(trainRowIds),
                  earlyStopRowIds = sort(earlyStopRowIds)
    )
    
  }
  
  return(datas)
}

convertToTorchData <- function(data, population, rowIds){
  x <- torch::torch_tensor(as.matrix(data[rowIds,]), dtype = torch::torch_float())
  
  #one-hot encoding
  y <- population$outcomeCount[population$rowId%in%rowIds]
  y[y>0] <- 1
  y <- torch::torch_tensor(matrix(y), dtype = torch::torch_float())
  
  return(list(x=x,
              y=y))
}

batchPredict <- function(model, 
                         plpData,
                         population,
                         predictRowIds,
                         batch_size ){
  ParallelLogger::logInfo('Predicting using  batch')
  maxVal <- length(predictRowIds)
  batches <- lapply(1:ceiling(maxVal/batch_size), function(x) ((x-1)*batch_size+1):min((x*batch_size), maxVal))
  
  ParallelLogger::logInfo('Pop')
  prediction <- population[population$rowId%in%predictRowIds,]
  prediction$value <- 0
  
  for(batch in batches){
    b <- torch::torch_tensor(as.matrix(plpData[predictRowIds[batch],, drop = F]), dtype = torch::torch_float())
    pred <- model(b)
    prediction$value[batch] <- as.array(pred$to())[,1]
  }
  attr(prediction, "metaData") <- list(predictionType = "binary")
  return(prediction)
}

#' @description converts a sparse Matrix into a list of its columns, 
#' subsequently vapply can be used to apply functions over the list 
listCols<-function(m){
    res<-split(m@x, findInterval(seq_len(Matrix::nnzero(m)), m@p, left.open=TRUE))
    names(res)<-colnames(m)
    res
}
  
  

