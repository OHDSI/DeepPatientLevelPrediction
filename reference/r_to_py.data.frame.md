# Use polars instead of pandas for default conversion from R to Python

Use polars instead of pandas for default conversion from R to Python

## Usage

``` r
# S3 method for class 'data.frame'
r_to_py(x, convert = FALSE)
```

## Arguments

- x:

  A data.frame object in R

- convert:

  Logical, whether to convert the data types to Python types

## Value

A polars DataFrame object in Python
