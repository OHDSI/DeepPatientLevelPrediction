#' converts a sparse Matrix into a list of its columns, 
#' subsequently vapply can be used to apply functions over the list 
listCols<-function(m){
    res<-split(m@x, findInterval(seq_len(length(m@x)), m@p, left.open=TRUE))
    names(res)<-colnames(m)
    res
}
  
  

