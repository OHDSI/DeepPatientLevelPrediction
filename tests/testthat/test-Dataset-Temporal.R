# Copyright 2025 Observational Health Data Sciences and Informatics
#
# This file is part of DeepPatientLevelPrediction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

createMockTemporalData <- function() {
  analysisRef <- data.frame(analysisId = 1, isBinary = "Y", missingMeansZero = "N")
  covariateRef <- data.frame(
    covariateId = 1:5,
    columnId = 1:5,
    covariateName = paste0("Cov_", 1:5),
    analysisId = 1
  )

  # Patient 1: Short sequence (3 events)
  # Patient 2: Long sequence (5 events) that needs truncation
  covariates <- data.frame(
    rowId = c(1, 1, 1, 2, 2, 2, 2, 2),
    covariateId = c(101, 102, 103, 201, 202, 203, 204, 205),
    columnId = c(1L, 2L, 3L, 1L, 2L, 3L, 4L, 5L),
    covariateValue = c(1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5),
    timeId = c(10, 20, 30, 5, 15, 25, 35, 45)
  )

  timeRef <- data.frame(timeId = 1:50)

  labels <- data.frame(
    rowId = 1:2,
    outcomeCount = c(1, 0)
  )

  andromedaData <- Andromeda::andromeda(
    analysisRef = analysisRef,
    covariateRef = covariateRef,
    covariates = covariates,
    timeRef = timeRef
  )

  return(list(data = andromedaData, labels = labels))
}


test_that("correctly truncates sequences longer than max_sequence_length", {
  mockData <- createMockTemporalData()

  dataset <- createDataset(
    data = mockData$data,
    labels = mockData$labels,
    temporalSettings = list(
      maxSequenceLength = 4L, # Force truncation for Patient 2
      timeTokens = FALSE
    )
  )
  featureIds <- dataset$data$feature_ids$numpy()
  expect_equal(featureIds[1, ], c(1, 2, 3, 0)) # padded to 4 events
  expect_equal(featureIds[2, ], c(1, 2, 3, 4)) # truncated to 4 events

  featureValues <- dataset$data$feature_values$numpy()
  expect_equal(featureValues[1, ], c(1.1, 1.2, 1.3, 0.0), tolerance = 1e-6) # Padded value is 0.0
  expect_equal(featureValues[2, ], c(2.1, 2.2, 2.3, 2.4), tolerance = 1e-6)

  seqLengths <- dataset$data$sequence_lengths$numpy()
  expect_equal(as.numeric(seqLengths), c(3, 4))
})

test_that("correctly pads sequences shorter than max_sequence_length", {
  mockData <- createMockTemporalData()

  dataset <- createDataset(
    data = mockData$data,
    labels = mockData$labels,
    temporalSettings = list(
      maxSequenceLength = 5L,
      timeTokens = FALSE
    )
  )

  featureIds <- dataset$data$feature_ids$numpy()
  expect_equal(featureIds[1, ], c(1, 2, 3, 0, 0))
})

test_that("time_ids are correctly prefixed and padded", {
  mockData <- createMockTemporalData()

  dataset <- createDataset(
    data = mockData$data,
    labels = mockData$labels,
    temporalSettings = list(
      maxSequenceLength = 4L,
      timeTokens = FALSE
    )
  )

  timeIds <- dataset$data$time_ids$numpy()

  expect_equal(ncol(timeIds), 5)
  expect_equal(timeIds[1, ], c(0, 10, 20, 30, 0))
  expect_equal(timeIds[2, ], c(0, 5, 15, 25, 35))
})

test_that("max_sequence_length = 'max' works correctly", {
  mockData <- createMockTemporalData()

  # The longest sequence in the mock data has 5 events
  dataset <- createDataset(
    data = mockData$data,
    labels = mockData$labels,
    temporalSettings = list(
      maxSequenceLength = "max",
      timeTokens = FALSE
    )
  )

  # The calculated max length should be 5
  expect_equal(dataset$max_sequence_length, 5)

  # Check shapes
  expect_equal(ncol(dataset$data$feature_ids$numpy()), 5)
  expect_equal(ncol(dataset$data$time_ids$numpy()), 6) # 5 + 1 for CLS
})

test_that("insert_time_tokens adds tokens correctly", {
  mockData <- createMockTemporalData()

  datasetWithTokens <- createDataset(
    data = mockData$data,
    labels = mockData$labels,
    temporalSettings = list(
      maxSequenceLength = 10L,
      timeTokens = TRUE
    )
  )

  featureInfo <- datasetWithTokens$get_feature_info()
  vocabSize <- featureInfo$get_vocabulary_size()

  offset <- 6

  # Patient 1 had time gaps of (20-10)=10 and (30-20)=10 days.
  # 10 days is in the W1 bucket (delta // 7), so token_number = 1.
  # Expected feature_id = offset + 1 = 7.
  # The original feature IDs were 1, 2, 3.
  # The sequence should be: [event 1, time_token, event 2, time_token, event 3]
  featureIdsP1 <- datasetWithTokens$data$feature_ids$numpy()[1, ]

  timeTokensP1 <- featureIdsP1[featureIdsP1 >= offset]

  expect_equal(featureIdsP1[1:5], c(1, 2, 7, 3, 7))
  expect_equal(length(timeTokensP1), 2)
  expect_true(all(timeTokensP1 == 7))
})
