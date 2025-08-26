FROM docker.io/rocker/r-ver:4.5 AS build

ENV MAKEFLAGS="-j4"
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lists,target=/var/lib/apt/lists,sharing=locked \
  apt-get -y update && apt-get install -y \
  default-jre \
  default-jdk \
  libssl-dev  \
  python3-dev \
  libpcre2-dev \
  libdeflate-dev \
  liblzma-dev \
  libbz2-dev \
  libicu-dev \
  xz-utils \
  libcurl4-openssl-dev \
  curl \
  ca-certificates \
  libxml2-dev \
  libpng-dev \
  cmake \
  libfontconfig1-dev \
  libgit2-dev \
  libharfbuzz-dev \
  libfribidi-dev \
  libfreetype6-dev \
  libtiff5-dev \
  libjpeg-dev \
  --no-install-recommends

RUN R CMD javareconf

RUN echo 'options(repos = c(CRAN = "https://p3m.dev/cran/__linux__/noble/latest"))' >>"${R_HOME}/etc/Rprofile.site"
ENV PAK_CACHE_DIR=/root/.cache/R/pak
RUN --mount=type=cache,id=pak-cache,target=/root/.cache/R/pak,sharing=locked \
  Rscript -e "install.packages('pak')"
RUN --mount=type=cache,id=pak-cache,target=/root/.cache/R/pak,sharing=locked \
  Rscript -e "pak::pkg_install(c('remotes', 'CirceR', 'Eunomia', 'duckdb', 'testthat', 'ResultModelManager'))"

RUN Rscript -e "DatabaseConnector::downloadJdbcDrivers(dbms='all', pathToDriver='/database_drivers/')"
ENV DATABASECONNECTOR_JAR_FOLDER=/database_drivers/

ADD https://astral.sh/uv/0.8.13/install.sh /uv-installer.sh
RUN sh /uv-installer.sh \
  && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/uv,sharing=locked \
  uv pip install --system --break-system-packages \
  polars \
  duckdb \
  pyarrow \
  numpy \
  torch \
  tqdm \
  pynvml

RUN --mount=type=secret,id=build_github_pat export GITHUB_PAT=$(cat /run/secrets/build_github_pat)
ARG GIT_BRANCH='main'
ARG GIT_COMMIT_ID_ABBREV
RUN Rscript -e "ref <- Sys.getenv('GIT_COMMIT_ID_ABBREV', unset = Sys.getenv('GIT_BRANCH')); remotes::install_github('ohdsi/DeepPatientLevelPrediction', ref=ref)" \
  && installGithub.r OHDSI/ROhdsiWebApi


FROM docker.io/rocker/rstudio:4.5
#
COPY --from=build /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=build /database_drivers /database_drivers
COPY --from=build /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=build /usr/local/lib/R/library /usr/local/lib/R/library

ENV RETICULATE_PYTHON=/usr/bin/python3
# runtime dependancies

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lists,target=/var/lib/apt/lists,sharing=locked \
  apt-get -y update && apt-get install -y \
  default-jre \
  default-jdk \
  libssl3 \
  python3-dev \
  libcurl4-openssl-dev \
  libpcre2-8-0 \
  libdeflate0 \
  liblzma5 \
  libbz2-1.0 \
  libicu74 \
  qpdf \
  --no-install-recommends \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && R CMD javareconf
