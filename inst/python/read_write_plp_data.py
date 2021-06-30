#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reading and writing Andromeda objects directly from Python
Andromeda objects that are based on a SQLite database can be read and written
directly from Python to avoid loading data through R."""
# --------------------------------------------------------------------------- #
#                  MODULE HISTORY                                             #
# --------------------------------------------------------------------------- #
# Version          1
# Date             2021-06-29
# Author           LH John
# Note             Original version
#
# --------------------------------------------------------------------------- #
#                  SYSTEM IMPORTS                                             #
# --------------------------------------------------------------------------- #
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------- #
#                  OTHER IMPORTS                                              #
# --------------------------------------------------------------------------- #
import sqlalchemy
import pandas as pd
import pyreadr

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  META DATA                                                  #
# --------------------------------------------------------------------------- #
__version__ = '1'
__status__ = 'Development'

# --------------------------------------------------------------------------- #
#                  CONSTANTS                                                  #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  GLOBAL VARIABLES                                           #
# --------------------------------------------------------------------------- #
# sqlite database engine
db_engine = None

# data read from .RDS files
outcomes = None
cohorts = None
meta_data = None
time_ref = None
cov_rds = None


# --------------------------------------------------------------------------- #
#                  EXPORTED FUNCTIONS                                         #
# --------------------------------------------------------------------------- #
def load_plp_data(path):
    """Load covariate data from an Andromeda object
    :param path: Path to Andromeda object
    :return: Covariate data
    """
    global db_engine
    sqlite_path = __uncompress_covariates(Path(path, "covariates"))
    __create_sqlite_engine(sqlite_path)
    # __test_sqlite_connection()
    __read_rds_files(path, sqlite_path)
    print(outcomes[None])
    covariate_ref = pd.read_sql('select * from covariateRef', db_engine)

    return covariate_ref


# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #
def __uncompress_covariates(path_to_covariates):
    """Uncompress SQLite database within an Andromeda object
    :return: Path to unzipped SQLite database holding covariate data"""
    sqlite_path = Path(Path.cwd(), "plpDataTemp")
    with zipfile.ZipFile(path_to_covariates, 'r') as zip_ref:
        zip_ref.extractall(sqlite_path)
    return sqlite_path


def __create_sqlite_engine(sqlite_path):
    """Create the database engine to query SQLite database
    :param sqlite_path: Path to uncompressed folder holding the SQLite database
    :return:
    """
    global db_engine
    rel_sqlite_db_path = list(Path(sqlite_path).glob("*.sqlite"))
    db_engine = sqlalchemy.create_engine(
        'sqlite:///' + str(rel_sqlite_db_path[0]))


def __test_sqlite_connection():
    """Just a testing function, which will be expanded or removed later
    :return:
    """
    # check tables: ['analysisRef', 'covariateRef', 'covariates']
    global db_engine
    inspector = sqlalchemy.inspect(db_engine)
    inspect_tables = inspector.get_table_names()
    print('Tables: %s' % inspect_tables)

    # Read data with pandas
    print(pd.read_sql('select * from covariateRef', db_engine))


def __read_rds_files(path, sqlite_path):
    """Read .RDS files from Andromeda object
    :param path: Path to Andromeda object
    :param sqlite_path: Path to uncompressed folder holding the SQLite database
    :return:
    """
    global outcomes, cohorts, meta_data, time_ref, cov_rds

    try:
        outcomes = pyreadr.read_r(Path(path, "outcomes.rds"))
    except pyreadr.custom_errors.LibrdataError:
        print("Andromeda file outcomes.rds could not be read.")

    try:
        cohorts = pyreadr.read_r(Path(path, "cohorts.rds"))
    except pyreadr.custom_errors.LibrdataError:
        print("Andromeda file cohorts.rds could not be read.")

    try:
        meta_data = pyreadr.read_r(Path(path, "metaData.rds"))
    except pyreadr.custom_errors.LibrdataError:
        print("Andromeda file metaData.rds could not be read.")

    try:
        time_ref = pyreadr.read_r(Path(path, "timeRef.rds"))
    except pyreadr.custom_errors.LibrdataError:
        print("Andromeda file timeRef.rds could not be read.")

    try:
        cov_rds = pyreadr.read_r(str(list(Path(sqlite_path).glob("*.rds"))[0]))
    except pyreadr.custom_errors.LibrdataError:
        print("Andromeda file outcomes.rds could not be read.")

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
