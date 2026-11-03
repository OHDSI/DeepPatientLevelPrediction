# PyTorch Module

The `torch` module object is the equivalent of
`reticulate::import("torch")` and is provided mainly as a convenience.
Accessing the module initializes Python and requires the Python
dependencies listed in `SystemRequirements` in the package `DESCRIPTION`
file.

## Format

An object of class `python.builtin.module`

## Value

The `torch` Python module.

## Examples

``` r
if (FALSE) { # \dontrun{
# Requires the Python dependencies described in vignette("Installing").
torch$randn(10L)
} # }
```
