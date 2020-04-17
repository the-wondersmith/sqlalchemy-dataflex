from sqlalchemy.dialects import registry as _registry

from .base import dfChar, dfDate, dfInt, dfDecimal

import pyodbc

__version__ = "0.0.1"

pyodbc.pooling = False  # Unchanged from SQLAlchemy-Access, doesn't seem to break anything

_registry.register(
    "dataflex.pyodbc", "sqlalchemy_dataflex.pyodbc", "DataFlexDialect_pyodbc"
)
