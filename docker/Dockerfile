FROM docker.io/rocker/r-ver:4.2.3 as build

RUN apt-get -y update && apt-get install -y \
      default-jre \
      default-jdk \
      python3-pip \
      python3-venv \
      libssl-dev  \
      --no-install-recommends \
      && apt-get clean

RUN echo "options(repos=list(CRAN='https://packagemanager.posit.co/cran/__linux__/jammy/latest'))" >> /root/.Rprofile

RUN R cmd javareconf

RUN Rscript -e "install.packages(c('remotes', 'testthat', 'Eunomia', 'CirceR', 'devtools'))" \ 
  && Rscript -e "remotes::install_github(c('ohdsi/CohortGenerator', 'ohdsi/ROhdsiWebApi'))"

RUN Rscript -e "remotes::install_github('ohdsi/DeepPatientLevelPrediction')"

RUN Rscript -e "DatabaseConnector::downloadJdbcDrivers(dbms='all', pathToDriver='/database_drivers/')"
ENV DATABASECONNECTOR_JAR_FOLDER=/database_drivers/

# set up python
ENV RETICULATE_MINICONDA_PATH=/opt/conda/
RUN Rscript -e "reticulate::install_miniconda()" \
    -e "reticulate::conda_install(envname = 'r-reticulate', packages=c('python=3.11'))"

# install Python packages
RUN /opt/conda/bin/conda run -n r-reticulate pip install connectorx polars pyarrow torch tqdm pynvml

# without conda and pip caches
ENV PIP_NO_CACHE_DIR=1
RUN rm -rf /root/.cache/R/renv \
    && /opt/conda/bin/conda clean -afy

FROM docker.io/rocker/rstudio:4.2.3

COPY --from=build /opt/conda /opt/conda
COPY --from=build /database_drivers /database_drivers
COPY --from=build /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=build /usr/local/lib/R/library /usr/local/lib/R/library

ENV RETICULATE_MINICONDA_PATH=/opt/conda/

# runtime dependanceis
RUN apt-get -y update && apt-get install -y \
      default-jre \
      libssl3 \
      --no-install-recommends \
      && R cmd javareconf \
      && apt-get clean

