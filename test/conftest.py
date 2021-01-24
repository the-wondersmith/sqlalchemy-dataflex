"""PyTest configuration for SQLAlchemy-Dataflex."""
# coding=utf-8

# noinspection PyPackageRequirements
import pytest
from sqlalchemy.dialects import registry


registry.register("dataflex.pyodbc", "sqlalchemy_dataflex.pyodbc", "DataflexDialect_pyodbc")

pytest.register_assert_rewrite("sqlalchemy.testing.assertions")

from sqlalchemy.testing.plugin.pytestplugin import *
