from sqlalchemy.dialects import registry as _registry

from .base import dfChar, dfDate, dfInt, dfDecimal

import pyodbc

__version__ = "0.0.2"

pyodbc.pooling = False  # required for DataFlex databases with ODBC linked tables
_registry.register(
    "dataflex.pyodbc", "sqlalchemy_dataflex.pyodbc", "DataFlexDialect_pyodbc"
)
