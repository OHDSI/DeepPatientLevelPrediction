# @file PackageMaintenance
#
# Copyright 2022 Observational Health Data Sciences and Informatics
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

# recreate the html index when new documentation
pkgdown::build_site()
OhdsiRTools::fixHadesLogo()

# Format and check code
OhdsiRTools::formatRFolder()
OhdsiRTools::checkUsagePackage("DeepPatientLevelPrediction")
OhdsiRTools::updateCopyrightYearFolder()
devtools::spell_check()

# Create manual and vignettes
unlink("extras/DeepPatientLevelPrediction.pdf")
system("R CMD Rd2pdf ./ --output=extras/DeepPatientLevelPrediction.pdf")

rmarkdown::render("vignettes/BuildingDeepModels.Rmd",
                  output_file = "../inst/doc/BuildingDeepModels.pdf",
                  rmarkdown::pdf_document(latex_engine = "pdflatex",
                                          toc = TRUE,
                                          toc_depth = 3,
                                          number_sections = TRUE))
