"""SQLAlchemy support for Dataflex databases."""
# coding=utf-8

import pyodbc
from sqlalchemy.dialects import registry as _registry

from .base import (
    nc,
    cg,
    Char,
    Date,
    Time,
    cl_in,
    BigInt,
    Double,
    Binary,
    Decimal,
    Integer,
    Logical,
    Numeric,
    VarChar,
    strtobool,
    Timestamp,
    LongVarChar,
    LongVarBinary,
    DoublePrecision,
)

__version__ = "0.1.2"

pyodbc.pooling = True  # Makes the ODBC overhead a little more manageable
_registry.register("dataflex.pyodbc", "sqlalchemy_dataflex.pyodbc", "DataflexDialect_pyodbc")

__all__ = (
    "nc",
    "cg",
    "Char",
    "Date",
    "Time",
    "cl_in",
    "BigInt",
    "Double",
    "Binary",
    "Decimal",
    "Logical",
    "Integer",
    "Numeric",
    "VarChar",
    "strtobool",
    "Timestamp",
    "__version__",
    "LongVarChar",
    "LongVarBinary",
    "DoublePrecision",
)
