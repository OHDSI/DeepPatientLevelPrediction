FROM docker.io/rocker/r-ver:4.4.1 AS build

RUN --mount=type=secret,id=build_github_pat export GITHUB_PAT=$(cat /run/secrets/build_github_pat)

ARG GIT_BRANCH='main'
ARG GIT_COMMIT_ID_ABBREV
ARG ARCH='amd64'


# optimize compilation
ENV MAKEFLAGS="-j$(2)"

RUN apt-get -y update && apt-get install -y \
      default-jre \
      default-jdk \
      libssl-dev  \
      python3-pip \
      python3-dev \
      libxml2-dev \
      libicu-dev \
      libbz2-dev \
      liblzma-dev \
      libdeflate-dev \
      libpcre2-dev \
      libcurl4-openssl-dev \
      --no-install-recommends \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*
RUN R CMD javareconf


RUN Rscript -e "install.packages('pak', \
                                 repos = sprintf('https://r-lib.github.io/p/pak/stable/%s/%s/%s', \
                                 'source', 'linux-gnu', if (Sys.getenv('ARCH') == 'amd64') 'amd64' else 'aarch64'))"

ENV DEBUGME=pkgdepends

RUN Rscript -e "options('repos'=c(RHUB='https://raw.githubusercontent.com/r-hub/repos/main/ubuntu-22.04-aarch64/4.4', \
                                   PPM='https://p3m.dev/cran/__linux__/jammy/latest')); \
     pak::pkg_install(c('remotes', \
                        'CirceR', \
                        'Eunomia', \
                        'duckdb', \
                        'DatabaseConnector', \
                        'ohdsi/CohortGenerator', \
                        'ohdsi/ROhdsiWebApi'))"

RUN Rscript -e "DatabaseConnector::downloadJdbcDrivers(dbms='all', pathToDriver='/database_drivers/')"
ENV DATABASECONNECTOR_JAR_FOLDER=/database_drivers/

# install Python packages
RUN pip3 install uv \
    && uv pip install --system --no-cache-dir \
    connectorx \
    polars \
    pyarrow \
    torch \
    tqdm \
    pynvml \
    && rm -rf /root/.cache/pip

RUN Rscript -e "ref <- Sys.getenv('GIT_COMMIT_ID_ABBREV', unset = Sys.getenv('GIT_BRANCH')); remotes::install_github('ohdsi/DeepPatientLevelPrediction', ref=ref)"


FROM docker.io/rocker/rstudio:4.4.1
#
COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=build /database_drivers /database_drivers
COPY --from=build /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=build /usr/local/lib/R/library /usr/local/lib/R/library

ENV RETICULATE_PYTHON=/usr/bin/python3
# runtime dependanceis
RUN apt-get -y update && apt-get install -y \
      default-jre \
      default-jdk \
      libssl3 \
      python3-dev \
      libxml2 \
      libicu70 \
      libbz2-1.0 \
      liblzma5 \
      libdeflate0 \
      libpcre2-8-0 \
      libcurl4 \
      --no-install-recommends \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/* \
      && R CMD javareconf

