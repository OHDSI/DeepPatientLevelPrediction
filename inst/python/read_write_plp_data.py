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
import tempfile

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

# --------------------------------------------------------------------------- #
#                  CLASS DEFINITION                                           #
# --------------------------------------------------------------------------- #
class ReadWritePlpData:
    """Read and write class for a single Andromeda data object."""

    def __init__(self):
        """Initialise instance variables to None"""
        self.db_engine = None
        self.outcomes = None
        self.cohorts = None
        self.meta_data = None
        self.time_ref = None
        self.cov_rds = None
        self.population = None
        self.tf = tempfile.TemporaryDirectory()

# --------------------------------------------------------------------------- #
#                  EXPORTED FUNCTIONS                                         #
# --------------------------------------------------------------------------- #
    def load_plp_data(self, path):
        """Load covariate data from an Andromeda object
        :param path: Path to Andromeda object
        :return:
        """
        self.__uncompress_covariates(path)
        self.__create_sqlite_engine(self.tf.name)
        self.__read_rds_files(path, self.tf.name)

    def load_population(self, path):
        """Load population from .RDS object
        :param path: Path to population .RDS object
        :return:
        """
        print("Loading population .RDS..")
        try:
            self.population = (pyreadr.read_r(str(Path(path))))[None]
        except pyreadr.custom_errors.LibrdataError as err:
            print("Handling LibrdataError: Population could not be read.", err)

    def get_population(self):
        """Returns population
        :return: Population
        """
        if self.population is not None:
            return self.population
        warnings.warn("No population loaded. First run load_population().")
        return None

    def get_covariates(self):
        """Returns covariates through SQLite query
        :return: Pandas DataFrame holding covariates
        """
        if self.db_engine is not None:
            covariates = pd.read_sql('select * from covariates',
                                     self.db_engine)
            return covariates
        warnings.warn("No connection to SQLite. First run load_plp_data().")
        return None

    def get_covariate_ref(self):
        """Returns covariateRef through SQLite query
        :return: Pandas DataFrame holding covariateRef
        """
        if self.db_engine is not None:
            covariate_ref = pd.read_sql('select * from covariateRef',
                                        self.db_engine)
            return covariate_ref
        warnings.warn("No connection to SQLite. First run load_plp_data().")
        return None

    def get_analysis_ref(self):
        """Returns analysisRef through SQLite query
        :return: Pandas DataFrame holding analysisRef
        """
        if self.db_engine is not None:
            analysis_ref = pd.read_sql('select * from analysisRef',
                                       self.db_engine)
            return analysis_ref
        warnings.warn("No connection to SQLite. First srun load_plp_data().")
        return None

    def custom_query(self, query):
        """Custom query the covariates, covariateRef or analysisRef table
        :param query: A string containing the SQL query
        :return: Query results as Pandas DataFrame
        """
        try:
            query_result = pd.read_sql(query, self.db_engine)
            return query_result
        except sqlalchemy.exc.ObjectNotExecutableError as err:
            print("Handling ObjectNotExecutableError: SQL query cannot be" +
                  " executed.", err)
            return None

    def get_tables(self):
        """Returns tables present in Andromeda SQLite database
        :return:
        """
        inspector = sqlalchemy.inspect(self.db_engine)
        inspect_tables = inspector.get_table_names()
        print('Tables: %s' % inspect_tables)
        return inspect_tables

# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #
    def __uncompress_covariates(self, path):
        """Uncompress SQLite database within an Andromeda object
        :param path: Path to Andromeda object
        :return:
        """
        print("Uncompressing data files..")
        path_to_covariates = Path(path, "covariates")
        with zipfile.ZipFile(path_to_covariates, 'r') as zip_ref:
            zip_ref.extractall(self.tf.name)

    def __create_sqlite_engine(self, sqlite_path):
        """Create the database engine to query SQLite database
        :param sqlite_path: Path to uncompressed folder with SQLite database
        :return:
        """
        print("Creating SQLite engine..")
        rel_sqlite_db_path = list(Path(sqlite_path).glob("*.sqlite"))
        self.db_engine = sqlalchemy.create_engine(
            'sqlite:///' + str(rel_sqlite_db_path[0]))

    def __read_rds_files(self, path, sqlite_path):
        """Read .RDS files from Andromeda object
        :param path: Path to Andromeda object
        :param sqlite_path: Path to uncompressed folder with SQLite database
        :return:
        """
        print("Reading .RDS files..")

        try:
            self.outcomes = pyreadr.read_r(str(Path(path,
                                                    "outcomes.rds")))[None]
        except pyreadr.custom_errors.LibrdataError as err:
            print("Handling LibrdataError: Andromeda file outcomes.rds could" +
                  " not be read.", err)

        try:
            self.cohorts = pyreadr.read_r(str(Path(path, "cohorts.rds")))[None]
        except pyreadr.custom_errors.LibrdataError as err:
            print("Handling LibrdataError: Andromeda file cohorts.rds could" +
                  " not be read.", err)

        try:
            self.meta_data = pyreadr.read_r(str(Path(path, "metaData.rds")))
        except pyreadr.custom_errors.LibrdataError as err:
            print("Handling LibrdataError: Andromeda file metaData.rds could" +
                  " not be read.", err)

        try:
            self.time_ref = pyreadr.read_r(str(Path(path, "timeRef.rds")))
        except pyreadr.custom_errors.LibrdataError as err:
            print("Handling LibrdataError: Andromeda file timeRef.rds could" +
                  " not be read.", err)

        try:
            self.cov_rds = pyreadr.read_r(str(list(Path(sqlite_path)
                                                   .glob("*.rds"))[0]))
        except pyreadr.custom_errors.LibrdataError as err:
            print("Handling LibrdataError: Andromeda file sqlite.rds could" +
                  " not be read.", err)

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
