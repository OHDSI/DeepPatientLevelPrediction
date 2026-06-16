# Expand a Component's Hyperparameter Grid

This function takes a user-defined setting for a configurable component
(like a positional encoding) and expands it into a flat list of all
possible configurations. This list can then be used directly within a
\`paramGrid\` for \`PatientLevelPrediction::listCartesian\`.

## Usage

``` r
expandComponentGrid(componentSetting)
```

## Arguments

- componentSetting:

  The user's setting for the component. This can be: - A single string
  (e.g., "AdamW"). - A single named list representing one configuration
  (e.g., \`list(name = "AdamW", lr = 0.001)\`). - A list of named lists,
  where each list is a template for configurations. Vectors within these
  templates will be expanded (e.g., \`lr = c(0.001, 0.0001)\`).

## Value

A list where each element is a fully-specified, named list for one
component configuration.
