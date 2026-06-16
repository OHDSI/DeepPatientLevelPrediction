testEnv <- environment()

closeAndromedaObject <- function(x, depth = 0L) {
  if (depth > 3L) {
    return(invisible())
  }
  if (inherits(x, "Andromeda")) {
    try(Andromeda::close(x), silent = TRUE)
    return(invisible())
  }
  if (is.list(x)) {
    for (item in x) {
      closeAndromedaObject(item, depth = depth + 1L)
    }
  }
  invisible()
}

for (name in ls(envir = testEnv, all.names = TRUE)) {
  try(closeAndromedaObject(get(name, envir = testEnv)), silent = TRUE)
}

objectsToRemove <- setdiff(
  ls(envir = testEnv, all.names = TRUE),
  c("testEnv", "closeAndromedaObject", "objectsToRemove")
)
rm(list = objectsToRemove, envir = testEnv)
invisible(gc())

if (
  requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = FALSE)
) {
  try(
    reticulate::py_run_string("
import gc

try:
    import torch
    if hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
except Exception:
    pass

gc.collect()
"),
    silent = TRUE
  )
}

invisible(gc())
