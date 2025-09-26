makeFakeTorch <- function(
  cudaAvailable = FALSE,
  cudaVersion = NULL,
  capMajor = 7L,
  capMinor = 5L,
  deviceName = "NVIDIA T4",
  bf16Supported = FALSE
) {
  list(
    cuda = list(
      is_available = function() cudaAvailable,
      get_device_capability = function() {
        list(as.integer(capMajor), as.integer(capMinor))
      },
      get_device_name = function() deviceName,
      is_bf16_supported = function() bf16Supported
    ),
    version = list(cuda = cudaVersion)
  )
}

localMockFlashBindings <- function(
  fakeTorch,
  flashModuleAvailable = FALSE,
  .scope = parent.frame()
) {
  # need to capture closure
  getTorchFn <- function() {
    fakeTorch
  }
  pyModuleAvailableFn <- function(mod) {
    if (identical(mod, "flash_attn.flash_attn_interface")) {
      return(flashModuleAvailable)
    }
    TRUE
  }
  testthat::local_mocked_bindings(
    getTorch = getTorchFn,
    pyModuleAvailable =  pyModuleAvailableFn,
    getSysName = function() "linux",
    .env = .scope
  )
  importStub <- function(mod, ...) {
    if (identical(mod, "flash_attn")) {
      structure(list(`__version__` = "9.9.9"), class = "py_module_stub")
    } else {
      reticulate::import(mod, ...)
    }
  }
  if (isTRUE(flashModuleAvailable)) {
      testthat::local_mocked_bindings(
        import = importStub,
        .package = "reticulate",
        .env = .scope
      )
    }
}
