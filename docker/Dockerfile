FROM docker.io/rocker/r-ver:4.2.3 as build

ARG GITHUB_PAT
ENV GITHUB_PAT=${GITHUB_PAT}

RUN apt-get -y update && apt-get install -y \
      default-jre \
      default-jdk \
      libxml2-dev \
      libglpk-dev \
      libsodium-dev \
      libsecret-1-dev \
      python3-pip \
      python3-venv \
      libssl-dev  \
      --no-install-recommends \
      && apt-get clean

WORKDIR /work_dir

RUN echo "options(repos=list(CRAN='https://packagemanager.posit.co/cran/__linux__/jammy/latest'))" >> /root/.Rprofile

COPY setupProjectEnv.R setupProjectEnv.R
RUN R cmd javareconf &&  \
    Rscript setupProjectEnv.R 

RUN Rscript -e "DatabaseConnector::downloadJdbcDrivers(dbms='all', pathToDriver='/database_drivers/')"

# set up python
ENV RETICULATE_MINICONDA_PATH=/opt/conda/
RUN Rscript -e "reticulate::install_miniconda()" \
    -e "reticulate::conda_install(envname = 'r-reticulate', packages=c('python=3.10'))"

# install Python packages
RUN /opt/conda/bin/conda run -n r-reticulate pip install connectorx polars pyarrow torch tqdm

# without conda and pip caches
ENV PIP_NO_CACHE_DIR=1
RUN rm -rf /root/.cache/R/renv \
    && /opt/conda/bin/conda clean -afy

FROM docker.io/rocker/r-ver:4.2.3

COPY --from=build /work_dir /work_dir
COPY --from=build /opt/conda /opt/conda
COPY --from=build /database_drivers /database_drivers
COPY --from=build /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=build /usr/local/lib/R/library /usr/local/lib/R/library

ENV RETICULATE_MINICONDA_PATH=/opt/conda/

# runtime dependanceis
RUN apt-get -y update && apt-get install -y \
      default-jre \
      libxml2 \
      libglpk40 \
      libsodium23 \
      libsecret-1-0 \
      libssl3 \
      --no-install-recommends \
      && R cmd javareconf \
      && apt-get clean

RUN mkdir /output

# create system keyring
WORKDIR /work_dir
RUN Rscript -e "install.packages('keyring')" && \
    Rscript -e "keyring::keyring_create('system', password='1234')"

ENV STRATEGUS_KEYRING_PASSWORD=1234
