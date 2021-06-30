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
import warnings

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
    :return:
    """
    sqlite_path = __uncompress_covariates(Path(path, "covariates"))
    __create_sqlite_engine(sqlite_path)
    __read_rds_files(path, sqlite_path)
    # __test_sqlite_connection()


def get_covariates():
    """Get covariates through SQLite query
    :return: Pandas DataFrame holding covariates
    """
    if db_engine is not None:
        covariates = pd.read_sql('select * from covariates', db_engine)
        return covariates
    warnings.warn("No connection to SQLite. First run load_plp_data().")
    return None


def get_covariate_ref():
    """Get covariateRef through SQLite query
    :return: Pandas DataFrame holding covariateRef
    """
    if db_engine is not None:
        covariate_ref = pd.read_sql('select * from covariateRef', db_engine)
        return covariate_ref
    warnings.warn("No connection to SQLite. First run load_plp_data().")
    return None


def get_analysis_ref():
    """Get analysisRef through SQLite query
    :return: Pandas DataFrame holding analysisRef
    """
    if db_engine is not None:
        analysis_ref = pd.read_sql('select * from analysisRef', db_engine)
        return analysis_ref
    warnings.warn("No connection to SQLite. First run load_plp_data().")
    return None


def custom_query(query):
    """Custom query the covariates, covariateRef or analysisRef table
    :param query: A string containing the SQL query
    :return: Query results as Pandas DataFrame
    """
    try:
        query_result = pd.read_sql(query, db_engine)
        return query_result
    except sqlalchemy.exc.ObjectNotExecutableError as err:
        print("Handling ObjectNotExecutableError: SQL query cannot be" +
              " executed.", err)
        return None


# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #
def __uncompress_covariates(path_to_covariates):
    """Uncompress SQLite database within an Andromeda object
    :return: Path to unzipped SQLite database holding covariate data"""
    print("Uncompressing data files..")
    sqlite_path = Path(Path.cwd(), "plpDataTemp")
    with zipfile.ZipFile(path_to_covariates, 'r') as zip_ref:
        zip_ref.extractall(sqlite_path)
    return sqlite_path


def __create_sqlite_engine(sqlite_path):
    """Create the database engine to query SQLite database
    :param sqlite_path: Path to uncompressed folder holding the SQLite database
    :return:
    """
    print("Creating SQLite engine..")
    global db_engine
    rel_sqlite_db_path = list(Path(sqlite_path).glob("*.sqlite"))
    db_engine = sqlalchemy.create_engine(
        'sqlite:///' + str(rel_sqlite_db_path[0]))


def __test_sqlite_connection():
    """Just a testing function, which will be expanded or removed later
    :return:
    """
    inspector = sqlalchemy.inspect(db_engine)
    inspect_tables = inspector.get_table_names()
    print('Tables: %s' % inspect_tables)
    print(outcomes[None])
    # Read data with pandas
    print(pd.read_sql('select * from covariateRef', db_engine))
    # print(outcomes.keys())
    # print(outcomes[None])

    # print(cohorts.keys())
    # print(cohorts[None])

    # print(time_ref.keys())
    # print(time_ref[None])

    # print(cov_rds.keys())
    # print(cov_rds[None])


def __read_rds_files(path, sqlite_path):
    """Read .RDS files from Andromeda object
    :param path: Path to Andromeda object
    :param sqlite_path: Path to uncompressed folder holding the SQLite database
    :return:
    """
    print("Reading .RDS files..")
    global outcomes, cohorts, meta_data, time_ref, cov_rds

    try:
        outcomes = pyreadr.read_r(Path(path, "outcomes.rds"))
    except pyreadr.custom_errors.LibrdataError as err:
        print("Handling LibrdataError: Andromeda file outcomes.rds could not" +
              " be read.", err)

    try:
        cohorts = pyreadr.read_r(Path(path, "cohorts.rds"))
    except pyreadr.custom_errors.LibrdataError as err:
        print("Handling LibrdataError: Andromeda file cohorts.rds could not" +
              " be read.", err)

    try:
        meta_data = pyreadr.read_r(Path(path, "metaData.rds"))
    except pyreadr.custom_errors.LibrdataError as err:
        print("Handling LibrdataError: Andromeda file metaData.rds could not" +
              " be read.", err)

    try:
        time_ref = pyreadr.read_r(Path(path, "timeRef.rds"))
    except pyreadr.custom_errors.LibrdataError as err:
        print("Handling LibrdataError: Andromeda file timeRef.rds could not" +
              " be read.", err)

    try:
        cov_rds = pyreadr.read_r(str(list(Path(sqlite_path).glob("*.rds"))[0]))
    except pyreadr.custom_errors.LibrdataError as err:
        print("Handling LibrdataError: Andromeda file outcomes.rds could not" +
              "be read.", err)

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
