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
db_engine = []


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
    __test_sqlite_connection()
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
    global db_engine
    rel_sqlite_db_path = list(Path(sqlite_path).glob("*.sqlite"))
    print(type(rel_sqlite_db_path[0]))
    db_engine = sqlalchemy.create_engine('sqlite:///'+str(rel_sqlite_db_path[0]))


def __test_sqlite_connection():
    # check tables: ['analysisRef', 'covariateRef', 'covariates']
    print(db_engine.table_names())
    # Read data with pandas
    print(pd.read_sql('select * from covariateRef', db_engine))

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
