library(PatientLevelPrediction)
sink(nullfile()) 

data(plpDataSimulationProfile)
plpData <- PatientLevelPrediction::simulatePlpData(plpDataSimulationProfile,
                                                   n=500)
# pseudo runPlp pipeline
populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
                          requireTimeAtRisk = F,
                          riskWindowStart = 1, 
                          riskWindowEnd = 365)
population <- PatientLevelPrediction::createStudyPopulation(plpData = plpData,
                                        outcomeId = 3,
                                        populationSettings = populationSettings)
data <- splitData(
      plpData = plpData,
      population = population,
      splitSettings = PatientLevelPrediction::createDefaultSplitSetting()
    )

trainData <- data$Train

sink()
