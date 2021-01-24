"""Run SQLAlchemy's dialect testing suite against the Dataflex dialect."""
# coding=utf-8

import sys
import pytest
import pyodbc
import operator
from pathlib import Path
from decimal import Decimal as PyDecimal
from datetime import date, time, datetime
from typing import Any, Dict, Union, Optional, Sequence as SequenceType

import sqlalchemy as sa
from itertools import chain
from sqlalchemy_dataflex import *
from sqlalchemy.testing import suite as sa_testing

from sqlalchemy.testing.suite import ComputedReflectionTest as _ComputedReflectionTest
from sqlalchemy.testing.suite import RowFetchTest as _RowFetchTest
from sqlalchemy.testing.suite import UnicodeVarcharTest as _UnicodeVarcharTest
from sqlalchemy.testing.suite import DateTimeCoercedToDateTimeTest as _DateTimeCoercedToDateTimeTest
from sqlalchemy.testing.suite import IsOrIsNotDistinctFromTest as _IsOrIsNotDistinctFromTest
from sqlalchemy.testing.suite import SimpleUpdateDeleteTest as _SimpleUpdateDeleteTest
from sqlalchemy.testing.suite import CollateTest as _CollateTest
from sqlalchemy.testing.suite import NormalizedNameTest as _NormalizedNameTest
from sqlalchemy.testing.suite import HasTableTest as _HasTableTest
from sqlalchemy.testing.suite import DateTimeHistoricTest as _DateTimeHistoricTest
from sqlalchemy.testing.suite import ComponentReflectionTest as _ComponentReflectionTest
from sqlalchemy.testing.suite import LastrowidTest as _LastrowidTest
from sqlalchemy.testing.suite import SequenceTest as _SequenceTest
from sqlalchemy.testing.suite import PercentSchemaNamesTest as _PercentSchemaNamesTest
from sqlalchemy.testing.suite import JSONStringCastIndexTest as _JSONStringCastIndexTest
from sqlalchemy.testing.suite import StringTest as _StringTest
from sqlalchemy.testing.suite import CompoundSelectTest as _CompoundSelectTest
from sqlalchemy.testing.suite import ReturningTest as _ReturningTest
from sqlalchemy.testing.suite import AutocommitTest as _AutocommitTest
from sqlalchemy.testing.suite import HasSequenceTest as _HasSequenceTest
from sqlalchemy.testing.suite import UnicodeTextTest as _UnicodeTextTest
from sqlalchemy.testing.suite import TimeMicrosecondsTest as _TimeMicrosecondsTest
from sqlalchemy.testing.suite import ExpandingBoundInTest as _ExpandingBoundInTest
from sqlalchemy.testing.suite import EscapingTest as _EscapingTest
from sqlalchemy.testing.suite import TableDDLTest as _TableDDLTest
from sqlalchemy.testing.suite import CompositeKeyReflectionTest as _CompositeKeyReflectionTest
from sqlalchemy.testing.suite import QuotedNameArgumentTest as _QuotedNameArgumentTest
from sqlalchemy.testing.suite import ExceptionTest as _ExceptionTest
from sqlalchemy.testing.suite import JSONTest as _JSONTest
from sqlalchemy.testing.suite import ExistsTest as _ExistsTest
from sqlalchemy.testing.suite import LimitOffsetTest as _LimitOffsetTest
from sqlalchemy.testing.suite import TimeTest as _TimeTest
from sqlalchemy.testing.suite import DateTest as _DateTest
from sqlalchemy.testing.suite import SequenceCompilerTest as _SequenceCompilerTest
from sqlalchemy.testing.suite import DateTimeMicrosecondsTest as _DateTimeMicrosecondsTest
from sqlalchemy.testing.suite import CTETest as _CTETest
from sqlalchemy.testing.suite import OrderByLabelTest as _OrderByLabelTest
from sqlalchemy.testing.suite import ComputedColumnTest as _ComputedColumnTest
from sqlalchemy.testing.suite import IntegerTest as _IntegerTest
from sqlalchemy.testing.suite import DateHistoricTest as _DateHistoricTest
from sqlalchemy.testing.suite import LikeFunctionsTest as _LikeFunctionsTest
from sqlalchemy.testing.suite import NumericTest as _NumericTest
from sqlalchemy.testing.suite import TimestampMicrosecondsTest as _TimestampMicrosecondsTest
from sqlalchemy.testing.suite import BooleanTest as _BooleanTest
from sqlalchemy.testing.suite import IsolationLevelTest as _IsolationLevelTest
from sqlalchemy.testing.suite import InsertBehaviorTest as _InsertBehaviorTest
from sqlalchemy.testing.suite import DateTimeTest as _DateTimeTest
from sqlalchemy.testing.suite import ServerSideCursorsTest as _ServerSideCursorsTest
from sqlalchemy.testing.suite import TextTest as _TextTest


def fix_filename(fname: Union[str, Path]) -> str:
    """Fix file paths."""
    if not isinstance(fname, str):
        fname = str(fname)

    if fname.casefold() == "unknown":
        return fname
    if cl_in("site-packages", fname):
        return "\\".join(fname.split("\\")[5:])
    if cl_in("PycharmProjects", fname):
        return "\\".join(fname.split("\\")[4:])

    raise ValueError(f"[{fname}] not a recognized file_path!")


def trace_calls(frame, event, arg):
    """Trace function calls."""
    if arg:
        del arg
    if event != "call":
        return
    co = frame.f_code
    func_name = co.co_name
    if any((cl_in(name, func_name) for name in ("write", "fix_filename", "__"))):
        # Ignore write() calls from print statements
        return
    func_name = func_name.replace("_", "\\_")
    func_line_no = frame.f_lineno
    func_filename = co.co_filename

    try:
        func_filename = fix_filename(func_filename)
    except (ValueError, TypeError, OSError):
        pass

    caller = frame.f_back
    caller_line_no = getattr(caller, "f_lineno", "unknown")
    caller_filename = getattr(getattr(caller, "f_code", None), "co_filename", "unknown")

    try:
        caller_filename = fix_filename(caller_filename)
    except (ValueError, TypeError, OSError):
        pass

    if not any((cl_in("sqlalchemy", func_filename), cl_in("sqlalchemy", caller_filename))):
        return

    log_line = f"| {func_name} | {func_line_no} | {func_filename} | {caller_line_no} | {caller_filename} |"
    with (Path().home() / "Downloads" / "trace.md").open("a+") as writer:
        writer.write(f"\n{log_line}")
        del writer


class DFTestTable:
    """A handy organizer for the static DataFlex tables required to test the dialect."""

    metadata: sa.MetaData
    schema: Optional[Any] = None

    def __init__(self, metadata: sa.MetaData, schema_=None) -> None:
        self.metadata = metadata
        self.schema = schema_

    def __bool__(self) -> bool:
        if getattr(self, "metadata", None) is not None:
            return True
        return False

    def _return_table(self, table: sa.Table) -> sa.Table:
        """Ensure that the supplied table is added to the MetaData instance."""
        self.metadata.create_all(tables=(table,))
        return table

    @property
    def _test_table(self) -> sa.Table:
        """The `_test_table` table."""
        return self._return_table(
            sa.Table(
                "_test_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("data", VarChar(50)),
                schema=self.schema,
            )
        )

    @property
    def autoinc_pk(self) -> sa.Table:
        """The `autoinc_pk` table."""
        return self._return_table(
            sa.Table(
                "autoinc_pk",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("data", VarChar(50)),
                schema=self.schema,
            )
        )

    @property
    def boolean_table(self) -> sa.Table:
        """The `boolean_table` table."""
        return self._return_table(
            sa.Table(
                "boolean_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("value", Logical),
                sa.Column("uc_value", Logical),
                schema=self.schema,
            )
        )

    @property
    def date_time_table(self) -> sa.Table:
        """The `date_time_table` table."""
        return self._return_table(
            sa.Table(
                "date_time_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("date_data", Timestamp),
                schema=self.schema,
            )
        )

    @property
    def date_table(self) -> sa.Table:
        """The `date_table` table."""
        return self._return_table(
            sa.Table(
                "date_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("date_data", Date),
                schema=self.schema,
            )
        )

    @property
    def integer_table(self) -> sa.Table:
        """The `integer_table` table."""
        return self._return_table(
            sa.Table(
                "integer_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("integer_data", Integer),
                schema=self.schema,
            )
        )

    @property
    def manual_pk(self) -> sa.Table:
        """The `manual_pk` table."""
        return self._return_table(
            sa.Table(
                "manual_pk",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("data", VarChar(50)),
                schema=self.schema,
            )
        )

    @property
    def plain_pk(self) -> sa.Table:
        """The `plain_pk` table."""
        return self._return_table(
            sa.Table(
                "plain_pk",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("data", VarChar(50)),
                schema=self.schema,
            )
        )

    @property
    def related(self) -> sa.Table:
        """The `related` table."""
        return self._return_table(
            sa.Table(
                "related",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("related", Integer),
                schema=self.schema,
            )
        )

    @property
    def some_int_table(self) -> sa.Table:
        """The `some_int_table` table."""
        return self._return_table(
            sa.Table(
                "some_int_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("x", Integer),
                sa.Column("y", Integer),
                schema=self.schema,
            )
        )

    @property
    def some_table(self) -> sa.Table:
        """The `some_table` table."""
        return self._return_table(
            sa.Table(
                "some_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("q", VarChar(50)),
                sa.Column("p", VarChar(50)),
                sa.Column("x", Integer),
                sa.Column("y", Integer),
                sa.Column("z", VarChar(50)),
                sa.Column("data", VarChar(100)),
                schema=self.schema,
            )
        )

    @property
    def stuff(self) -> sa.Table:
        """The `stuff` table."""
        return self._return_table(
            sa.Table(
                "stuff",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("data", VarChar(50)),
                schema=self.schema,
            )
        )

    @property
    def tb1(self) -> sa.Table:
        """The `tb1` table."""
        return self._return_table(
            sa.Table(
                "tb1",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("attr", Integer),
                sa.Column("name", VarChar(20)),
                schema=self.schema,
            )
        )

    @property
    def tb2(self) -> sa.Table:
        """The `tb2` table."""
        return self._return_table(
            sa.Table(
                "tb2",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("pid", Integer),
                sa.Column("pattr", Integer),
                sa.Column("pname", VarChar(20)),
                ForeignKeyConstraint(
                    ["pname", "pid", "pattr"],
                    [self.tb1.c.name, self.tb1.c.id, self.tb1.c.attr],
                    name="fk_tb1_name_id_attr",
                ),
                schema=self.schema,
            )
        )

    @property
    def test_table(self) -> sa.Table:
        """The `test_table` table."""
        return self._return_table(
            sa.Table(
                "test_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("data", VarChar(50)),
                schema=self.schema,
            )
        )

    @property
    def unicode_table(self) -> sa.Table:
        """The `unicode_table` table."""
        return self._return_table(
            sa.Table(
                "unicode_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("unicode_data", VarChar(255)),
                schema=self.schema,
            )
        )

    @property
    def users(self) -> sa.Table:
        """The `users` table."""
        return self._return_table(
            sa.Table(
                "users",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("test1", VarChar(5)),
                sa.Column("test2", Double),
                schema=self.schema,
            )
        )

    @property
    def foo(self) -> sa.Table:
        """The `foo` table."""
        return self._return_table(sa.Table("foo", self.metadata, sa.Column("one", VarChar), schema=self.schema))

    @property
    def float_table(self) -> sa.Table:
        """The `float_table` table."""
        return self._return_table(sa.Table("float_table", self.metadata, sa.Column("x", Double), schema=self.schema))

    @property
    def varchar_table(self) -> sa.Table:
        """The `varchar_table` table."""
        return self._return_table(
            sa.Table("varchar_table", self.metadata, sa.Column("x", VarChar), schema=self.schema)
        )

    @property
    def numeric_table(self) -> sa.Table:
        """The `numeric_table` table."""
        return self._return_table(
            sa.Table("numeric_table", self.metadata, sa.Column("x", sa.Numeric), schema=self.schema)
        )

    @property
    def int_table(self) -> sa.Table:
        """The `int_table` table."""
        return self._return_table(sa.Table("int_table", self.metadata, sa.Column("x", Integer), schema=self.schema))

    @property
    def dingalings(self) -> sa.Table:
        """The `dingalings` table."""
        return self._return_table(
            sa.Table(
                "dingalings",
                self.metadata,
                sa.Column("dingaling_id", Integer, primary_key=True, nullable=True),
                sa.Column("address_id", Integer, sa.ForeignKey("email_addresses.address_id")),
                sa.Column("data", VarChar(30)),
                schema=self.schema,
            )
        )

    @property
    def email_addresses(self) -> sa.Table:
        """The `email_addresses` table."""
        return self._return_table(
            sa.Table(
                "email_addresses",
                self.metadata,
                sa.Column("address_id", Integer),
                sa.Column("remote_user_id", Integer, sa.ForeignKey("users.id")),
                sa.Column("email_address", VarChar(20)),
                sa.PrimaryKeyConstraint("address_id", name="email_ad_pk"),
                schema=self.schema,
            )
        )

    @property
    def comment_test(self) -> sa.Table:
        """The `comment_test` table."""
        return self._return_table(
            sa.Table(
                "comment_test",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True, comment="id comment"),
                sa.Column("data", VarChar(20), comment="data % comment"),
                sa.Column("d2", VarChar(20), comment=r"""Comment types type speedily ' " \ '' Fun!""",),
                schema=self.schema,
                comment=r"""the test % ' " \ table comment""",
            )
        )

    @property
    def noncol_idx_test_nopk(self) -> sa.Table:
        """The `noncol_idx_test_nopk` table."""
        return self._return_table(
            sa.Table("noncol_idx_test_nopk", self.metadata, sa.Column("q", VarChar(5)), schema=self.schema,)
        )

    @property
    def noncol_idx_test_pk(self) -> sa.Table:
        """The `noncol_idx_test_pk` table."""
        return self._return_table(
            sa.Table(
                "noncol_idx_test_pk",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True),
                sa.Column("q", VarChar(5)),
                schema=self.schema,
            )
        )

    @property
    def testtbl(self) -> sa.Table:
        """The `testtbl` table."""
        return self._return_table(
            sa.Table(
                "testtbl",
                self.metadata,
                sa.Column("a", VarChar(20)),
                sa.Column("b", VarChar(30)),
                sa.Column("c", Integer),
                # reserved identifiers
                sa.Column("asc", VarChar(30)),
                sa.Column("key", VarChar(30)),
                schema=self.schema,
            )
        )

    @property
    def types_table(self) -> sa.Table:
        """The `types_table` table."""
        return self._return_table(
            sa.Table(
                "types_table",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True, autoincrement=False),
                sa.Column("ASCII", VarChar(255)),
                sa.Column("BIGINT", BigInt),
                sa.Column("BINARY", LongVarBinary(2624)),
                sa.Column("BOOLEAN", Logical),
                sa.Column("CHAR", Char(255)),
                sa.Column("DATE", Date),
                sa.Column("DECIMAL", Decimal),
                sa.Column("DOUBLE", DoublePrecision),
                sa.Column("FLOAT", DoublePrecision),
                sa.Column("INTEGER", Integer),
                sa.Column("LOGICAL", Logical),
                sa.Column("LONGVARBINARY", LongVarBinary(2624)),
                sa.Column("LONGVARCHAR", LongVarChar(2624)),
                sa.Column("NUMERIC", Decimal),
                sa.Column("TEXT", LongVarChar(2624)),
                sa.Column("TIME", Time),
                sa.Column("TIMESTAMP", Timestamp),
                sa.Column("VARBINARY", LongVarBinary(2624)),
                sa.Column("VARCHAR", LongVarChar(2624)),
                schema=self.schema,
            )
        )

    @property
    def t(self) -> sa.Table:
        """The `t` table."""
        return self._return_table(
            sa.Table(
                "t",
                self.metadata,
                # This may or may not work correctly,
                # as it's unclear if DataFlex cares about
                # nullability
                sa.Column("a", Integer, nullable=True),
                sa.Column("b", Integer, nullable=False),
                sa.Column("data", VarChar(50), nullable=True),
                schema=self.schema,
            )
        )

    @property
    def includes_defaults(self) -> sa.Table:
        """The `includes_defaults` table."""
        return self._return_table(
            sa.Table(
                "includes_defaults",
                self.metadata,
                sa.Column("id", Integer, primary_key=True, nullable=True, autoincrement=True),
                sa.Column("data", VarChar(50)),
                sa.Column("x", Integer, default=5),
                sa.Column("y", Integer, default=sa.literal_column("2", type_=Integer) + sa.literal(2),),
                schema=self.schema,
            )
        )

    @property
    def scalar_select(self) -> sa.Table:
        """The `scalar_select` table."""
        return self._return_table(
            sa.Table(
                "scalar_select", self.metadata, sa.Column("data", VarChar(50), nullable=True), schema=self.schema,
            )
        )


def df_literal_round_trip(self, type_, input_, output, filter_=None):
    """Test literal value rendering."""

    # The SQLAlchemy test suite usually creates a table for this
    # test on the fly. The FlexODBC driver can't do that though,
    # so we'll have to use a specially created table with one
    # column of each type, named accordingly.

    column_name = getattr(
        type_, "__type_name__", getattr(type_, "__visit_name__", getattr(type_, "__dataflex_name__", "VARCHAR"))
    ).upper()

    # For literals, we test the literal render in an INSERT
    # into a typed column.  We can then SELECT it back as its
    # "official" type.
    t = DFTestTable(getattr(self, "metadata", sa.MetaData())).types_table

    t.delete().execute()

    with sa_testing.testing.db.connect() as conn:
        for pos, value in enumerate(input_):
            statement = (
                t.insert()
                .values(**{"id": pos + 1, column_name: value})
                .compile(dialect=sa_testing.testing.db.dialect, compile_kwargs=dict(literal_binds=True))
            )
            conn.execute(statement)

        if self.supports_whereclause:
            stmt = sa.select([t.c.id, getattr(t.c, column_name, t.c.VARCHAR)]).where(
                getattr(t.c, column_name, t.c.VARCHAR) == value
            )
        else:
            stmt = sa.select([t.c.id, getattr(t.c, column_name, t.c.VARCHAR)])

        stmt = stmt.compile(dialect=sa_testing.testing.db.dialect, compile_kwargs=dict(literal_binds=True))
        for row in conn.execute(stmt):
            value = row[1]
            if filter_ is not None:
                value = filter_(value)
            assert value in output


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class AutocommitTest(_AutocommitTest):
    """Test the dialect's handling of autocommit."""


class BooleanTest(_BooleanTest):
    """Test the dialect's handling of boolean values."""

    __backend__ = True

    @classmethod
    def define_tables(cls, metadata: sa.MetaData):
        """Define the table(s) required by the test(s)."""
        assert DFTestTable(metadata).boolean_table is not None

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    def test_render_literal_bool(self):
        self._literal_round_trip(Logical, [True, False], [True, False])

    def test_round_trip(self):
        boolean_table = self.tables.boolean_table

        sa_testing.config.db.execute(boolean_table.insert({"id": 1, "value": True, "uc_value": False}))

        row = sa_testing.config.db.execute(sa.select([boolean_table.c.value, boolean_table.c.uc_value])).first()

        sa_testing.eq_(row, (True, False))
        assert isinstance(row[0], bool)

    def test_whereclause(self):
        # testing "WHERE <column>" renders a compatible expression
        boolean_table = self.tables.boolean_table

        with sa_testing.config.db.connect() as conn:
            conn.execute(boolean_table.insert({"id": 1, "value": True, "uc_value": True}))
            conn.execute(boolean_table.insert({"id": 2, "value": False, "uc_value": False}))

            sa_testing.eq_(
                conn.scalar(sa.select([boolean_table.c.id]).where(boolean_table.c.value)), 1,
            )
            sa_testing.eq_(
                conn.scalar(sa.select([boolean_table.c.id]).where(boolean_table.c.uc_value)), 1,
            )
            sa_testing.eq_(
                conn.scalar(sa.select([boolean_table.c.id]).where(~boolean_table.c.value)), 2,
            )
            sa_testing.eq_(
                conn.scalar(sa.select([boolean_table.c.id]).where(~boolean_table.c.uc_value)), 2,
            )


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class CTETest(_CTETest):
    """Test the dialect's handling of CTE."""


class CollateTest(_CollateTest):
    """Test the dialect's handling of collation."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required by the test(s)."""
        assert DFTestTable(metadata).some_table is not None

    @classmethod
    def insert_data(cls, connection):
        """Insert the data required by the test(s)."""
        table = cls.tables.some_table

        inserts = (
            table.insert({"id": 1, "data": "collate data1"}),
            table.insert({"id": 2, "data": "collate data2"}),
        )

        for query in inserts:
            connection.execute(query)

    @sa_testing.testing.requires.order_by_collation
    def test_collate_order_by(self):
        collation = sa_testing.testing.requires.get_order_by_collation(sa_testing.testing.config)
        table = self.tables.some_table

        query = sa.select([table.c.id, table.c.data]).order_by(table.c.data.collate(collation).asc())
        expected = [(1, "collate data1"), (2, "collate data2")]

        result = sa_testing.config.db.execute(query).fetchall()

        sa_testing.eq_(result, expected)


class ComponentReflectionTest(_ComponentReflectionTest):
    """Test the dialect's handling of component reflection."""

    @classmethod
    def define_reflected_tables(cls, metadata, schema_):
        """Define the tables required by the reflection test(s)."""

        df_tables = DFTestTable(metadata, schema_)

        for table in (
            "users",
            "dingalings",
            "comment_test",
            "email_addresses",
            "noncol_idx_test_pk",
            "noncol_idx_test_nopk",
        ):
            assert getattr(df_tables, table, None) is not None

    @sa_testing.testing.provide_metadata
    def _type_round_trip(self, *types):
        df_tables = DFTestTable(self.metadata)
        assert df_tables.types_table is not None

        type_names = set(
            chain.from_iterable(
                (
                    filter(
                        None,
                        (
                            f"""{getattr(item, f"__{name}_name__", None)}({getattr(item, "length", None)})"""
                            if getattr(item, "length", None)
                            else getattr(item, f"__{name}_name__", None)
                            for name in ("dataflex", "type", "visit")
                        ),
                    )
                )
                for item in types
            )
        )

        def type_filter(item: Dict[str, Any]) -> bool:
            """Filter out requested types."""
            item_type = item.get("type", None)

            if item_type is None:
                return False

            item_type_names = set(
                filter(
                    None,
                    (
                        f"""{getattr(item_type, f"__{name}_name__", None)}({getattr(item_type, "length", None)})"""
                        if getattr(item_type, "length", None)
                        else getattr(item_type, f"__{name}_name__", None)
                        for name in ("dataflex", "type", "visit")
                    ),
                )
            )

            return bool(type_names.intersection(item_type_names))

        return list(
            filter(
                None,
                (
                    c.get("type", None)
                    for c in filter(type_filter, sa.inspect(self.metadata.bind).get_columns("types_table"))
                ),
            )
        )

    @sa_testing.testing.requires.table_reflection
    @sa_testing.testing.provide_metadata
    def test_autoincrement_col(self):
        """test that 'autoincrement' is reflected according to sqla's policy.

        Don't mark this test as unsupported for any backend !

        (technically it fails with MySQL InnoDB since "id" comes before "id2")

        A backend is better off not returning "autoincrement" at all,
        instead of potentially returning "False" for an auto-incrementing
        primary key column.
        """

        meta = self.metadata
        insp = sa.inspect(meta.bind)

        for tname, cname in [
            ("users", "id"),
            ("email_addresses", "id"),
            ("dingalings", "id"),
        ]:
            cols = insp.get_columns(tname)
            id_ = {c["name"]: c for c in cols}[cname]
            assert id_.get("autoincrement", True)

    @pytest.mark.skip(reason="DataFlex check constraint support currently unclear.")
    def test_get_check_constraints(self):
        """DocString."""

    @sa_testing.testing.provide_metadata
    def _test_get_columns(self, schema_=None, table_type="table"):
        """Test column reflection."""

        df_tables = DFTestTable(self.metadata, schema_)
        users, addresses = (df_tables.users, df_tables.email_addresses)
        table_names = ["users", "email_addresses"]
        insp = sa.inspect(self.metadata.bind)

        for table_name, table in zip(table_names, (users, addresses)):
            schema_name = sa_testing.schema
            cols = insp.get_columns(table_name, schema=schema_name)
            self.assert_(len(cols) > 0, len(cols))

            # should be in order

            expected_data = {"name", "type", "nullable", "default", "autoincrement"}

            for i, col in enumerate(table.columns):
                assert cols[i]["name"] in col.name

                # The built=in SQLAlchemy test checks for a whole bunch
                # of other things, most importantly that the column types
                # match up as expected. Because of DataFlex's limited
                # set of column types, that's... not feasible. So we'll
                # reduce the SQLAlchemy check to simply ensuring that
                # the dialect's `get_columns` method returned the expected
                # information

                difference = expected_data.difference(set(cols[i].keys()))

                self.assert_(len(difference) == 0, f"Missing column data: {difference}")

                if not col.primary_key:
                    assert cols[i]["default"] is None

    @sa_testing.testing.provide_metadata
    def _test_get_indexes(self, schema_=None):
        """DocString."""
        meta = self.metadata

        # The database may decide to create indexes for foreign keys, etc.
        # so there may be more indexes than expected.
        insp = sa.inspect(meta.bind)
        indexes = insp.get_indexes("users", schema=schema_)
        expected_indexes = [
            {"name": "INDEX2", "unique": True, "column_names": ["test1", "test2"]},
            {"name": "INDEX1", "unique": True, "column_names": ["id", "test2", "test1"]},
        ]
        self._assert_insp_indexes(indexes, expected_indexes)

    @pytest.mark.skip(reason="FlexODBC driver doesn't support primary key reflection.")
    def test_get_pk_constraint(self):
        """DocString."""
        pass

    @sa_testing.testing.provide_metadata
    def _test_get_noncol_index(self, tname, ixname):
        # tname = "noncol_idx_test_nopk"
        # ixname = "noncol_idx_nopk"

        meta = self.metadata
        insp = sa.inspect(meta.bind)
        indexes = insp.get_indexes(tname)

        # reflecting an index that has "x DESC" in it as the column.
        # the DB may or may not give us "x", but make sure we get the index
        # back, it has a name, it's connected to the table.
        expected_indexes = [{"unique": True, "name": ixname}]

        self._assert_insp_indexes(indexes, expected_indexes)

        t = sa.Table(tname, meta, autoload_with=meta.bind)
        sa_testing.eq_(len(t.indexes), 1)
        assert list(t.indexes)[0].table is t
        sa_testing.eq_(list(t.indexes)[0].name, ixname)

    @sa_testing.testing.requires.index_reflection
    @sa_testing.testing.requires.indexes_with_ascdesc
    def test_get_noncol_index_no_pk(self):
        self._test_get_noncol_index("noncol_idx_test_nopk", "INDEX1")

    @sa_testing.testing.requires.index_reflection
    @sa_testing.testing.requires.indexes_with_ascdesc
    def test_get_noncol_index_pk(self):
        self._test_get_noncol_index("noncol_idx_test_pk", "INDEX1")

    @sa_testing.testing.provide_metadata
    def _test_get_table_names(self, schema_=None, table_type="table", order_by=None):
        _ignore_tables = [
            "comment_test",
            "noncol_idx_test_pk",
            "noncol_idx_test_nopk",
            "local_table",
            "remote_table",
            "remote_table_2",
        ]
        meta = self.metadata

        insp = sa.inspect(meta.bind)

        tables = insp.get_table_names(schema_)
        table_names = [t for t in tables if t not in _ignore_tables]

        answer = ["dingalings", "email_addresses", "users"]
        for table_name in answer:
            assert table_name in table_names

    @sa_testing.testing.provide_metadata
    def _test_get_unique_constraints(self, schema_=None):
        # DataFlex doesn't allow for named indexes, it enforces
        # its own naming scheme (INDEX + sequential number).
        uniques = sorted(
            [
                {"name": "INDEX1", "unique": True, "column_names": ["a"]},
                {"name": "INDEX2", "unique": True, "column_names": ["a", "b", "c"]},
                {"name": "INDEX3", "unique": True, "column_names": ["c", "a", "b"]},
                {"name": "INDEX4", "unique": True, "column_names": ["asc", "key"]},
                {"name": "INDEX5", "unique": True, "column_names": ["b"]},
                {"name": "INDEX6", "unique": True, "column_names": ["c"]},
            ],
            key=operator.itemgetter("name"),
        )
        orig_meta = self.metadata
        assert DFTestTable(orig_meta, schema_).testtbl is not None

        inspector = sa.inspect(orig_meta.bind)
        reflected = sorted(
            inspector.get_unique_constraints("testtbl", schema=schema_), key=operator.itemgetter("name"),
        )

        names_that_duplicate_index = set()

        for orig, refl in zip(uniques, reflected):
            # Different dialects handle duplicate index and constraints
            # differently, so ignore this flag
            dupe = refl.pop("duplicates_index", None)
            if dupe:
                names_that_duplicate_index.add(dupe)
            sa_testing.eq_(orig, refl)

        reflected_metadata = sa.MetaData()
        reflected = sa.Table("testtbl", reflected_metadata, autoload_with=orig_meta.bind, schema=schema_)

        # Test to ensure that the reflected index names are exactly what
        # we expected, nothing more or less
        reflected_names = {idx.name for idx in reflected.indexes}
        expected_names = {idx.get("name") for idx in uniques}

        assert len(expected_names.difference(reflected_names)) == 0
        assert len(reflected_names.difference(expected_names)) == 0
        assert reflected_names == expected_names

    @sa_testing.testing.requires.table_reflection
    @sa_testing.testing.provide_metadata
    def test_nullable_reflection(self):
        assert DFTestTable(self.metadata).t is not None

        columns = sa.inspect(self.metadata.bind).get_columns("t")

        # Due to the way that DataFlex does its indexing, and the fact that DataFlex
        # tables are discrete flat-files, all columns are technically nullable
        expected = {"a": True, "b": True, "data": True}
        result = dict((col["name"], col["nullable"]) for col in columns)

        sa_testing.eq_(result, expected)

    @sa_testing.testing.requires.table_reflection
    def test_varchar_reflection(self):
        round_trip_types = self._type_round_trip(VarChar(255))
        typ = round_trip_types[0]
        assert isinstance(typ, (VarChar, LongVarChar))
        sa_testing.eq_(typ.length, 255)


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class CompositeKeyReflectionTest(_CompositeKeyReflectionTest):
    """Test the dialect's handling of composite key reflection."""

    # The FlexODBC driver doesn't support primary or foreign key reflection


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class CompoundSelectTest(_CompoundSelectTest):
    """Test the dialect's handling of compound select clauses."""

    # FlexODBC driver doesn't support UNION, JOIN or any other form of compound select.


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class ComputedColumnTest(_ComputedColumnTest):
    """Test the dialect's handling of computed columns."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class ComputedReflectionTest(_ComputedReflectionTest):
    """Test the dialect's handling of computed reflection."""


class DateHistoricTest(_DateHistoricTest):
    """Test the dialect's handling of historic date data."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.date_table is not None

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    def test_null(self):
        date_table = self.tables.date_table

        sa_testing.config.db.execute(date_table.insert({"id": 1, "date_data": None}))

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 1)).first()
        sa_testing.eq_(row, (None,))

    def test_round_trip(self):
        date_table = self.tables.date_table

        sa_testing.config.db.execute(date_table.insert({"id": 2, "date_data": self.data}))

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 2)).first()

        compare = self.compare or self.data
        sa_testing.eq_(row, (compare,))
        assert isinstance(row[0], type(compare))

    @sa_testing.testing.requires.datetime_literals
    def test_literal(self):
        compare = self.compare or self.data
        self._literal_round_trip(Date, [self.data], [compare])


class DateTest(_DateTest):
    """Test the dialect's handling of date data."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.date_table is not None

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    def test_null(self):
        date_table = self.tables.date_table

        sa_testing.config.db.execute(date_table.insert({"id": 1, "date_data": None}))

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 1)).first()
        sa_testing.eq_(row, (None,))

    def test_round_trip(self):
        date_table = self.tables.date_table

        sa_testing.config.db.execute(date_table.insert({"id": 2, "date_data": self.data}))

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 2)).first()

        compare = self.compare or self.data
        sa_testing.eq_(row, (compare,))
        assert isinstance(row[0], type(compare))

    @sa_testing.testing.requires.datetime_literals
    def test_literal(self):
        compare = self.compare or self.data
        self._literal_round_trip(Date, [self.data], [compare])


class DateTimeCoercedToDateTimeTest(_DateTimeCoercedToDateTimeTest):
    """Test the dialect's ability to coerce datetime data to datetime objects."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.date_time_table is not None

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    def test_null(self):
        date_table = self.tables.date_time_table

        sa_testing.config.db.execute(date_table.insert({"id": 1, "date_data": None}))

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 1)).first()
        sa_testing.eq_(row, (None,))

    def test_round_trip(self):
        date_table = self.tables.date_time_table

        sa_testing.config.db.execute(date_table.insert({"id": 2, "date_data": self.data}))

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 2)).first()

        compare = self.data or self.compare
        sa_testing.eq_(row, (compare,))
        assert isinstance(row[0], type(compare))

    @sa_testing.testing.requires.datetime_literals
    def test_literal(self):
        compare = self.compare or self.data
        self._literal_round_trip(Timestamp, [self.data], [compare])


class DateTimeHistoricTest(_DateTimeHistoricTest):
    """Test the dialect's handling of historic datetime data."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.date_time_table is not None

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    def test_null(self):
        date_table = self.tables.date_time_table

        sa_testing.config.db.execute(date_table.insert({"id": 1, "date_data": None}))

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 1)).first()
        sa_testing.eq_(row, (None,))

    def test_round_trip(self):
        date_table = self.tables.date_time_table

        sa_testing.config.db.execute(date_table.insert({"id": 2, "date_data": self.data}))

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 2)).first()

        compare = self.compare or self.data
        sa_testing.eq_(row, (compare,))
        assert isinstance(row[0], type(compare))

    @sa_testing.testing.requires.datetime_literals
    def test_literal(self):
        compare = self.compare or self.data
        self._literal_round_trip(Timestamp, [self.data], [compare])


class DateTimeMicrosecondsTest(_DateTimeMicrosecondsTest):
    """Test the dialect's handling of datetime data with microseconds."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.date_time_table is not None

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    def test_null(self):
        date_table = self.tables.date_time_table

        sa_testing.config.db.execute(date_table.insert({"id": 1}))
        sa_testing.config.db.execute(
            date_table.update()
            .values(date_data=sa.text("NULL"))
            .where(date_table.c.id == 1)
            .compile(dialect=self.bind.dialect, compile_kwargs=dict(literal_binds=True))
        )

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 1)).first()
        sa_testing.eq_(row, (None,))

    def test_round_trip(self):
        date_table = self.tables.date_time_table

        sa_testing.config.db.execute(date_table.insert({"id": 2}))
        sa_testing.config.db.execute(
            date_table.update()
            .values(date_data=self.data)
            .where(date_table.c.id == 2)
            .compile(dialect=self.bind.dialect, compile_kwargs=dict(literal_binds=True))
        )

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 2)).first()

        compare = self.compare or self.data
        sa_testing.eq_(row, (compare,))
        assert isinstance(row[0], type(compare))

    @sa_testing.testing.requires.datetime_literals
    def test_literal(self):
        compare = self.compare or self.data
        self._literal_round_trip(Timestamp, [self.data], [compare])


class DateTimeTest(_DateTimeTest):
    """Test the dialect's handling of datetime data."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.date_time_table is not None

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    def test_null(self):
        date_table = self.tables.date_time_table

        sa_testing.config.db.execute(date_table.insert({"id": 1}))
        sa_testing.config.db.execute(
            date_table.update()
            .values(date_data=sa.text("NULL"))
            .where(date_table.c.id == 1)
            .compile(dialect=self.bind.dialect, compile_kwargs=dict(literal_binds=True))
        )

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 1)).first()
        sa_testing.eq_(row, (None,))

    def test_round_trip(self):
        date_table = self.tables.date_time_table

        sa_testing.config.db.execute(date_table.insert({"id": 2}))
        sa_testing.config.db.execute(
            date_table.update()
            .values(date_data=self.data)
            .where(date_table.c.id == 2)
            .compile(dialect=self.bind.dialect, compile_kwargs=dict(literal_binds=True))
        )

        row = sa_testing.config.db.execute(sa.select([date_table.c.date_data]).where(date_table.c.id == 2)).first()

        compare = self.compare or self.data
        sa_testing.eq_(row, (compare,))
        assert isinstance(row[0], type(compare))

    @sa_testing.testing.requires.datetime_literals
    def test_literal(self):
        compare = self.compare or self.data
        self._literal_round_trip(Timestamp, [self.data], [compare])


class EscapingTest(_EscapingTest):
    """Test the dialect's handling of escaping."""

    @sa_testing.testing.provide_metadata
    def test_percent_sign_round_trip(self):
        """test that the DBAPI accommodates for escaped / non-escaped
        percent signs in a way that matches the compiler

        """
        m = self.metadata
        df_tables = DFTestTable(m)
        t = df_tables.t

        with sa_testing.config.db.begin() as conn:
            sa_testing.config.db.execute(
                t.insert({"data": "some % value"}).compile(
                    dialect=m.bind.dialect, compile_kwargs=dict(literal_binds=True)
                )
            )
            sa_testing.config.db.execute(
                t.insert({"data": "some %% other value"}).compile(
                    dialect=m.bind.dialect, compile_kwargs=dict(literal_binds=True)
                )
            )

            sa_testing.eq_(
                conn.scalar(
                    sa.select([t.c.data])
                    .where(t.c.data == "some % value")
                    .compile(dialect=m.bind.dialect, compile_kwargs=dict(literal_binds=True))
                ),
                "some % value",
            )

            sa_testing.eq_(
                conn.scalar(
                    sa.select([t.c.data])
                    .where(t.c.data == "some %% other value")
                    .compile(dialect=m.bind.dialect, compile_kwargs=dict(literal_binds=True))
                ),
                "some %% other value",
            )


class ExceptionTest(_ExceptionTest):
    """Test the dialect's raising and handling of exceptions."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.manual_pk is not None

    @pytest.mark.skip(reason="Crashes FlexODBC driver.")
    def test_integrity_error(self):
        """Test that table integrity is maintained."""

        # Technically, this test *should* pass because the
        # FlexODBC driver *does* maintain table integrity,
        # it just does so by crashing so badly that it takes
        # the ODBC caller with it.

        # In order for record integrity to even matter to DataFlex
        # there has to be an index of the relevant column(s). Any
        # table without an index on columns that should be unique
        # won't raise any kind of error. If there *is* an index,
        # an error *should* be raise, but whether or not that
        # actually happens is unknown at the time of this writing.

    @sa_testing.testing.provide_metadata
    def test_exception_with_non_ascii(self):

        table = DFTestTable(self.metadata).some_table
        conn = self.bind.engine

        try:
            # Try to create an error message that contains non-ascii
            # characters in FlexODBC or pyodbc's error message string.
            query = sa.select([table.c.id, sa.literal_column("méil")])
            conn.execute(query).fetchall()
            assert False
        except Exception as err:
            err_str = f"{type(err)} -> {err}"

            assert str(err.orig) in str(err)

        assert isinstance(err_str, str)


class ExistsTest(_ExistsTest):
    """Test the dialect's handling of `exists` clauses."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.stuff is not None

    @classmethod
    def insert_data(cls, connection):
        """Ensure that any data required by the test(s) exists in the relevant table(s)."""
        for dataset in [
            {"id": 1, "data": "some data"},
            {"id": 2, "data": "some data"},
            {"id": 3, "data": "some data"},
            {"id": 4, "data": "some other data"},
        ]:
            connection.execute(cls.tables.stuff.insert(dataset))

    def test_select_exists(self, connection):
        stuff = self.tables.stuff
        query = (
            sa.select([sa.literal(1)])
            .where(sa.exists().where(stuff.c.data == "some data"))
            .compile(dialect=sa_testing.testing.db.dialect, compile_kwargs=dict(literal_binds=True))
        )
        result = connection.execute(query).fetchone()
        sa_testing.eq_(result, (1,))

    def test_select_exists_false(self, connection):
        stuff = self.tables.stuff
        query = (
            sa.select([sa.literal(1)])
            .where(sa.exists().where(stuff.c.data == "no data"))
            .compile(dialect=sa_testing.testing.db.dialect, compile_kwargs=dict(literal_binds=True))
        )
        result = connection.execute(query).fetchall()
        sa_testing.eq_(result, [])


class ExpandingBoundInTest(_ExpandingBoundInTest):
    """Test the dialect's handling of expanding bound in statements."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.some_table is not None

    @classmethod
    def insert_data(cls, connection):
        """Ensure that any data required by the test(s) exists in the relevant table(s)."""
        for dataset in [
            {"id": 1, "x": 1, "y": 2, "z": "z1"},
            {"id": 2, "x": 2, "y": 3, "z": "z2"},
            {"id": 3, "x": 3, "y": 4, "z": "z3"},
            {"id": 4, "x": 4, "y": 5, "z": "z4"},
        ]:
            connection.execute(cls.tables.some_table.insert(dataset))

    @pytest.mark.skip(reason="Proper implementation currently unknown.")
    def test_null_in_empty_set_is_false(self):

        # This test *should* result in the compiler instance emitting a query that looks
        # something like:
        #
        # SELECT CASE WHEN (NULL IN (SELECT 1 FROM (SELECT 1) WHERE 1!=1)) THEN 1 ELSE 0 END AS "anon_1"
        #
        # There are a few issues that prevent the creation of the applicable queries though.
        # First, FlexODBC requires all SELECT queries to include at least one valid FROM clause.
        # Second, FlexODBC doesn't support CASE or IFF or anything similar. It also doesn't
        # *technically* support the use of an empty set in an IN clause.
        # Lastly, it *appears* as though the version(s) of FlexODBC that are available at the
        # time of this writing don't handle sub-queries correctly when they're (mis)used for this
        # specific purpose. It acts as though the entire result set is NULL instead of returning
        # the expected result. To work around these issues, the dialect uses some relatively simple
        # logic to generate sub-queries that amount to the same thing as though a true empty set was
        # supplied and correctly parsed. For example:
        #
        # SELECT "some_table"."id" FROM "some_table"
        # WHERE "some_table"."x" IN (SELECT MAX("some_table"."x") + 1 FROM "some_table")
        # ORDER BY "some_table"."id"
        #
        # Thereby creating an unsatisfiable condition. This works for all of the tests in this
        # class, with the exception of this one. At the time of this writing, (with the sole possible
        # exception of the ANSI SQL function `IFNULL`, which FlexODBC *does* support) no workable
        # solution or workaround that could emulate the functionality of CASE or IFF has presented
        # itself. Until one is devised, this test is un-passable.

        # noinspection PyTypeChecker
        stmt = sa.select(
            [case([(null().in_(sa.bindparam("foo", value=(), expanding=True)), true(),)], else_=false(),)]
        )
        in_(sa_testing.config.db.execute(stmt).fetchone()[0], (False, 0))


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class HasSequenceTest(_HasSequenceTest):
    """Test the dialect's handling of has sequence statements."""


class HasTableTest(_HasTableTest):
    """Test the dialect's handling of table introspection."""


class InsertBehaviorTest(_InsertBehaviorTest):
    """Test the dialect's `insert` behavior."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        for table_name in (
            "autoinc_pk",
            "manual_pk",
            "includes_defaults",
        ):
            assert getattr(df_tables, table_name, None) is not None

    @pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
    def test_empty_insert(self):
        """Test the dialect's handling of empty insert statements."""
        pass

    @sa_testing.requirements.insert_from_select
    def test_insert_from_select(self):
        table: sa.Table = self.tables.manual_pk

        for param_dict in [{"id": 1, "data": "data1"}, {"id": 2, "data": "data2"}, {"id": 3, "data": "data3"}]:
            sa_testing.config.db.execute(table.insert(param_dict))

        orig_query = sa.select([table.c.id + 5, table.c.data]).where(table.c.data.in_(["data2", "data3"]))
        action = table.insert(inline=True).from_select(("id", "data"), orig_query)
        sa_testing.config.db.execute(action)

        check_query = sa.select([table.c.data]).order_by(table.c.data)
        result = sa_testing.config.db.execute(check_query).fetchall()

        sa_testing.eq_(result, [("data1",), ("data2",), ("data2",), ("data3",), ("data3",)])

    @sa_testing.requirements.insert_from_select
    def test_insert_from_select_autoinc(self):
        src_table: sa.Table = self.tables.manual_pk
        dest_table: sa.Table = self.tables.autoinc_pk

        for param_dict in [{"id": 1, "data": "data1"}, {"id": 2, "data": "data2"}, {"id": 3, "data": "data3"}]:
            sa_testing.config.db.execute(src_table.insert(param_dict))

        orig_query = sa.select([src_table.c.data]).where(src_table.c.data.in_(["data2", "data3"]))
        action = dest_table.insert().from_select(("data",), orig_query)

        result = sa_testing.config.db.execute(action)

        sa_testing.eq_(result.inserted_primary_key, [None])

        check_query = sa.select([dest_table.c.data]).order_by(dest_table.c.data)
        result = sa_testing.config.db.execute(check_query).fetchall()

        sa_testing.eq_(result, [("data2",), ("data3",)])

    @sa_testing.requirements.insert_from_select
    def test_insert_from_select_autoinc_no_rows(self):
        src_table: sa.Table = self.tables.manual_pk
        dest_table: sa.Table = self.tables.autoinc_pk

        orig_query = sa.select([src_table.c.data]).where(src_table.c.data.in_(["data2", "data3"]))
        action = dest_table.insert().from_select(("data",), orig_query)

        result = sa_testing.config.db.execute(action)

        sa_testing.eq_(result.inserted_primary_key, [None])

        check_query = sa.select([dest_table.c.data]).order_by(dest_table.c.data)
        result = sa_testing.config.db.execute(check_query).fetchall()

        sa_testing.eq_(result, [])

    @pytest.mark.skip(reason="FlexODBC driver doesn't (currently) like 'complex' insertion values.")
    @sa_testing.requirements.insert_from_select
    def test_insert_from_select_with_defaults(self):
        table = self.tables.includes_defaults

        for param_dict in [{"id": 1, "data": "data1"}, {"id": 2, "data": "data2"}, {"id": 3, "data": "data3"}]:
            sa_testing.config.db.execute(table.insert(param_dict))

        orig_query = sa.select([table.c.id + 5, table.c.data]).where(table.c.data.in_(["data2", "data3"]))
        action = table.insert(inline=True).from_select(("id", "data"), orig_query)

        sa_testing.config.db.execute(action)
        expected = [
            (1, "data1", 5, 4),
            (2, "data2", 5, 4),
            (7, "data2", 5, 4),
            (3, "data3", 5, 4),
            (8, "data3", 5, 4),
        ]
        check_query = sa.select([table]).order_by(table.c.data, table.c.id)
        result = sa_testing.config.db.execute(check_query).fetchall()

        sa_testing.eq_(result, expected)


class IntegerTest(_IntegerTest):
    """Test the dialect's handling of integer data."""

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    def test_literal(self):
        self._literal_round_trip(Integer, [5], [5])

    @sa_testing.testing.provide_metadata
    def test_huge_int(self):
        data = 1376537018368127
        table = DFTestTable(self.metadata).types_table

        sa_testing.config.db.execute(table.delete())

        statement = table.insert({"id": 1, "BIGINT": data})
        sa_testing.config.db.execute(statement)

        query = sa.select([table.c.BIGINT])
        row = sa_testing.config.db.execute(query).first()

        sa_testing.eq_(row, (data,))

        assert isinstance(row[0], int)


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class IsOrIsNotDistinctFromTest(_IsOrIsNotDistinctFromTest):
    """Test the dialect's handling of `is` or `is not` distinct from clauses."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class IsolationLevelTest(_IsolationLevelTest):
    """Test the dialect's handling of isolation levels."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class JSONStringCastIndexTest(_JSONStringCastIndexTest):
    """Test the dialect's handling of JSON data."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class JSONTest(_JSONTest):
    """Test the dialect's handling of JSON data."""


@pytest.mark.skip(reason="Currently no way to implement / emulate auto-increment behavior.")
class LastrowidTest(_LastrowidTest):
    """Test the dialect's ability to retrieve the id of the last inserted row."""

    # NOTE: DataFlex doesn't support or have an auto-incrementing column type.
    #       Also, FlexODBC doesn't support any of the functions that would
    #       make it possible to emulate that behavior. Until a workable method
    #       for doing so is identified, these tests will have to be skipped.

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        for table_name in (
            "autoinc_pk",
            "manual_pk",
        ):
            assert getattr(df_tables, table_name, None) is not None

    def _assert_round_trip(self, table, conn):
        row = conn.execute(table.select()).first()
        sa_testing.eq_(row, (sa_testing.config.db.dialect.default_sequence_base, "some data"))

    @sa_testing.testing.provide_metadata
    def test_autoincrement_on_insert(self):
        table = DFTestTable(self.metadata).autoinc_pk
        statement = table.insert({"data": "some data"})

        sa_testing.config.db.execute(statement)

        self._assert_round_trip(table, sa_testing.config.db)

    @sa_testing.testing.provide_metadata
    def test_last_inserted_id(self):
        table = DFTestTable(self.metadata).autoinc_pk
        statement = table.insert({"data": "some data"})

        r = sa_testing.config.db.execute(statement)
        pk = sa_testing.config.db.scalar(sa.select([table.c.id]))

        sa_testing.eq_(r.inserted_primary_key, [pk])


class LikeFunctionsTest(_LikeFunctionsTest):
    """Test the dialect's handling of `like` with functions."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required to run the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.some_table is not None

    @classmethod
    def insert_data(cls, connection):
        """Ensure that any data required by the test(s) exists in the relevant table(s)."""
        table = cls.tables.some_table
        for param_set in [
            {"id": 1, "data": "abcdefg"},
            {"id": 2, "data": "ab/cdefg"},
            {"id": 3, "data": "ab%cdefg"},
            {"id": 4, "data": "ab_cdefg"},
            {"id": 5, "data": "abcde/fg"},
            {"id": 6, "data": "abcde%fg"},
            {"id": 7, "data": "ab#cdefg"},
            {"id": 8, "data": "ab9cdefg"},
            {"id": 9, "data": "abcde#fg"},
            {"id": 10, "data": "abcd9fg"},
        ]:
            connection.execute(table.insert(param_set))

    def _test(self, expr, expected):
        some_table = self.tables.some_table

        with sa_testing.config.db.connect() as conn:
            rows = {value for value, in conn.execute(sa.select([some_table.c.id]).where(expr))}

        sa_testing.eq_(rows, expected)

    def test_contains_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains("b%cde", autoescape=True), {3})

    def test_contains_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains("b%cd", autoescape=True, escape="#"), {3})
        self._test(col.contains("b#cd", autoescape=True, escape="#"), {7})

    def test_contains_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.contains("b##cde", escape="#"), {7})

    def test_contains_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.contains("b%cde"), {1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_endswith_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith("e%fg", autoescape=True), {6})

    def test_endswith_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith("e%fg", autoescape=True, escape="#"), {6})
        self._test(col.endswith("e#fg", autoescape=True, escape="#"), {9})

    def test_endswith_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith("e##fg", escape="#"), {9})

    def test_endswith_sqlexpr(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith(sa.literal_column("'e%fg'")), {1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_endswith_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.endswith("e%fg"), {1, 2, 3, 4, 5, 6, 7, 8, 9})

    def test_startswith_autoescape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith("ab%c", autoescape=True), {3})

    def test_startswith_autoescape_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith("ab%c", autoescape=True, escape="#"), {3})
        self._test(col.startswith("ab#c", autoescape=True, escape="#"), {7})

    def test_startswith_escape(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith("ab##c", escape="#"), {7})

    def test_startswith_sqlexpr(self):
        col = self.tables.some_table.c.data
        self._test(
            col.startswith(sa.literal_column("'ab%c'")), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        )

    def test_startswith_unescaped(self):
        col = self.tables.some_table.c.data
        self._test(col.startswith("ab%c"), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})


class LimitOffsetTest(_LimitOffsetTest):
    """Test the dialect's handling of limit / offset clauses."""

    @classmethod
    def define_tables(cls, metadata):
        """Create the table(s) required by the test(s)."""
        df_tables = DFTestTable(metadata)
        assert df_tables.some_table is not None

    @classmethod
    def insert_data(cls, connection):
        """Ensure that any data required by the test(s) exits on the relevant table(s)."""
        connection.execute(cls.tables.some_table.delete())
        for param_set in [
            {"id": 1, "x": 1, "y": 2},
            {"id": 2, "x": 2, "y": 3},
            {"id": 3, "x": 3, "y": 4},
            {"id": 4, "x": 4, "y": 5},
        ]:
            connection.execute(cls.tables.some_table.insert(param_set))

    def _assert_result(self, select, result, params=()):
        sa_testing.eq_(sa_testing.config.db.execute(select, params).fetchall(), result)

    def test_simple_limit(self):
        table = self.tables.some_table

        query = sa.select([table.c.id, table.c.x, table.c.y]).order_by(table.c.id).limit(2)
        expected = [(1, 1, 2), (2, 2, 3)]

        self._assert_result(query, expected)

    @sa_testing.testing.provide_metadata
    @sa_testing.testing.requires.bound_limit_offset
    def test_bound_limit(self):
        table = self.tables.some_table

        query = sa.select([table.c.id, table.c.x, table.c.y]).order_by(table.c.id).limit(sa.bindparam("l"))
        expected = [(1, 1, 2), (2, 2, 3)]
        params = {"l": 2}

        self._assert_result(query, expected, params=params)

    @pytest.mark.skip("FlexODBC drover doesn't support `OFFSET`")
    @sa_testing.testing.requires.offset
    def test_simple_offset(self):
        table = self.tables.some_table

        query = sa.select([table.c.id, table.c.x, table.c.y]).order_by(table.c.id).offset(2)
        expected = [(3, 3, 4), (4, 4, 5)]

        self._assert_result(query, expected)

    @pytest.mark.skip("FlexODBC drover doesn't support `OFFSET`")
    @sa_testing.testing.requires.offset
    def test_simple_limit_offset(self):
        table = self.tables.some_table

        query = sa.select([table.c.id, table.c.x, table.c.y]).order_by(table.c.id).limit(2).offset(1)
        expected = [(2, 2, 3), (3, 3, 4)]

        self._assert_result(query, expected)

    @pytest.mark.skip("FlexODBC drover doesn't support `OFFSET`")
    @sa_testing.testing.requires.offset
    def test_limit_offset_nobinds(self):
        """test that 'literal binds' mode works - no bound params."""

        table = self.tables.some_table

        query = (
            sa.select([table.c.id, table.c.x, table.c.y])
            .order_by(table.c.id)
            .limit(2)
            .offset(1)
            .compile(dialect=sa_testing.config.db.dialect, compile_kwargs={"literal_binds": True})
            .string
        )
        expected = [(2, 2, 3), (3, 3, 4)]

        self._assert_result(query, expected)

    @pytest.mark.skip("FlexODBC drover doesn't support `OFFSET`")
    @sa_testing.testing.requires.bound_limit_offset
    def test_bound_offset(self):
        table = self.tables.some_table

        query = sa.select([table.c.id, table.c.x, table.c.y]).order_by(table.c.id).offset(sa.bindparam("o"))
        expected = [(3, 3, 4), (4, 4, 5)]
        params = {"o": 2}

        self._assert_result(query, expected, params=params)

    @pytest.mark.skip("FlexODBC drover doesn't support `OFFSET`")
    @sa_testing.testing.requires.bound_limit_offset
    def test_bound_limit_offset(self):
        table = self.tables.some_table

        query = (
            sa.select([table.c.id, table.c.x, table.c.y])
            .order_by(table.c.id)
            .limit(sa.bindparam("l"))
            .offset(sa.bindparam("o"))
        )
        expected = [(2, 2, 3), (3, 3, 4)]
        params = {"l": 2, "o": 1}

        self._assert_result(query, expected, params=params)


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class NormalizedNameTest(_NormalizedNameTest):
    """Test the dialect's handling of normalized names."""


class NumericTest(_NumericTest):
    """Test the dialect's handling of numeric data."""

    __scalar_table_ready = False

    @sa_testing.testing.provide_metadata
    def _scalar_table_ready(self):
        """Ensure there is at least one value in the `scalar_select` table."""
        if not self.__scalar_table_ready:
            metadata = self.metadata
            engine: sa.engine.base.Engine = metadata.bind
            table = DFTestTable(metadata).scalar_select

            query = sa.select([table.c.data]).limit(1)

            check = engine.execute(query).fetchall()
            if len(check) < 1:
                engine.execute(table.insert({"data": "I have approximate knowledge of many things."}))
                return self._scalar_table_ready()

            self.__scalar_table_ready = True
        return self.__scalar_table_ready

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """test literal rendering """

        col_name = getattr(
            type_, "__type_name__", getattr(type_, "__dataflex_name__", getattr(type_, "__visit_name__", "VARCHAR"))
        ).upper()

        metadata = self.metadata
        engine = metadata.bind
        table = sa.Table(
            "types_table",
            metadata,
            sa.Column("id", Integer, primary_key=True, nullable=True, autoincrement=False),
            sa.Column(col_name, type_),
        )

        engine.execute(table.delete())

        with sa_testing.testing.db.connect() as conn:
            for id_, value in enumerate(input_):
                ins = (
                    table.insert()
                    .values({"id": id_ + 1, col_name: sa.literal(value)})
                    .compile(dialect=sa_testing.testing.db.dialect, compile_kwargs=dict(literal_binds=True),)
                )
                conn.execute(ins)

            if self.supports_whereclause:
                stmt = sa.select([getattr(table.c, col_name)]).where(getattr(table.c, col_name) == sa.literal(value))
            else:
                stmt = sa.select([getattr(table.c, col_name)])

            stmt = stmt.compile(dialect=sa_testing.testing.db.dialect, compile_kwargs=dict(literal_binds=True),)
            for row in conn.execute(stmt):
                value = row[0]
                if filter_ is not None:
                    value = filter_(value)
                assert value in output

    @sa_testing.testing.provide_metadata
    def _get_scalar_value(self, value):
        """Emulate table-less SELECT queries."""

        # The FlexODBC driver doesn't like trying to select values
        # without a table to select from - it *will* raise an error.
        # It will, however, happily select scalar values from an
        # existing table, as long as there is at least one record
        # in that table. As such, we need to ensure that there's at
        # least one record in the table we're using for scalar selects
        # and we should be golden.

        metadata = self.metadata
        engine = metadata.bind
        df_tables = DFTestTable(metadata)
        table = df_tables.scalar_select

        assert self._scalar_table_ready()

        value_class = getattr(value, "value", value).__class__

        assert value_class in (str, int, bool, float, None.__class__, PyDecimal, bytes)

        result = engine.execute(
            sa.select(columns=[value], from_obj=table)
            .limit(1)
            .compile(dialect=self.metadata.bind.dialect, compile_kwargs={"literal_binds": True})
            .string
        ).fetchone()

        if result:
            return value_class(result[0])
        return None

    @sa_testing.testing.emits_warning(r".*does \*not\* support Decimal objects natively")
    @sa_testing.testing.provide_metadata
    def _do_test(self, type_, input_, output, filter_=None, check_scale=False, **kwargs):

        col_name = getattr(
            type_, "__type_name__", getattr(type_, "__dataflex_name__", getattr(type_, "__visit_name__", "VARCHAR"))
        ).upper()

        metadata = self.metadata
        engine = metadata.bind
        table = sa.Table(
            "types_table",
            metadata,
            sa.Column("id", Integer, primary_key=True, nullable=True, autoincrement=False),
            sa.Column(col_name, type_),
        )

        engine.execute(table.delete())

        for id_, value in enumerate(input_):
            engine.execute(table.insert({"id": id_ + 1, col_name: value}))

        query = sa.select([getattr(table.c, col_name).label("x")])

        result = {row[0] for row in engine.execute(query)}
        output = set(output)
        if filter_:
            result = set(map(filter_, result))
            output = set(map(filter_, output))
        sa_testing.eq_(result, output)
        if check_scale:
            sa_testing.eq_(list(map(str, result)), list(map(str, output)))

    @sa_testing.testing.requires.implicit_decimal_binds
    @sa_testing.testing.emits_warning(r".*does \*not\* support Decimal objects natively")
    @sa_testing.testing.provide_metadata
    def test_decimal_coerce_round_trip(self):
        expr = PyDecimal("15.7563")

        val = self._get_scalar_value(sa.literal(expr))
        sa_testing.eq_(val, expr)

    @pytest.mark.skip(reason="Emulation of CAST/CONVERT not currently implemented.")
    @sa_testing.testing.emits_warning(r".*does \*not\* support Decimal objects natively")
    def test_decimal_coerce_round_trip_w_cast(self):
        expr = PyDecimal("15.7563")

        # val = sa_testing.testing.db.scalar(sa.select([sa.cast(expr, Numeric(10, 4))]))
        val = self._get_scalar_value(sa.cast(expr, Numeric(10, 4)))
        sa_testing.eq_(val, expr)

    @sa_testing.testing.requires.floats_to_four_decimals
    def test_float_as_decimal(self):

        # The original SQLAlchemy test passes Decimal("15.7563") and
        # None as the expected output values for the test, however
        # the FlexODBC driver (or possibly just DataFlex in general)
        # won't allow NULL values to be stored. We *could* force
        # the conversion function in the dialect's Decimal type-class
        # to evaluate all 0.0000 decimals as None, but that would
        # mean we couldn't store 0-value decimals. The expected output
        # for the test has been modified accordingly.

        self._do_test(
            Numeric(precision=8, as_decimal=True),
            [15.7563, PyDecimal("15.7563"), None],
            [PyDecimal("15.7563"), PyDecimal("0.000000")],
        )

    def test_float_as_float(self):
        self._do_test(
            Numeric(precision=8, as_float=True, as_decimal=False),
            [15.7563, PyDecimal("15.7563")],
            [15.7563],
            filter_=lambda n: float(n) if (n is not None and round(n, 5) or None) else n,
        )

    @sa_testing.testing.provide_metadata
    def test_float_coerce_round_trip(self):
        expr = 15.7563

        val = self._get_scalar_value(sa.literal(expr))
        sa_testing.eq_(val, expr)

    @sa_testing.testing.requires.precision_generic_float_type
    def test_float_custom_scale(self):

        # The built-in SQLAlchemy test uses a return scale of 7,
        # but DataFlex (and therefore FlexODBC) only support a
        # scale up to 6. This test has been modified to reflect
        # that by reducing the return scale *below* that maximum
        # to ensure that it's being respected.

        self._do_test(
            Numeric(None, decimal_return_scale=5, as_decimal=True),
            [15.75638, PyDecimal("15.75638")],
            [PyDecimal("15.75638")],
            check_scale=True,
        )

    def test_numeric_as_float(self):
        self._do_test(
            Numeric(precision=8, scale=4, asdecimal=False),
            [15.7563, PyDecimal("15.7563")],
            [15.7563],
            filter_=lambda n: float(n) if n is not None else n,
        )

    @sa_testing.testing.requires.fetch_null_from_numeric
    def test_numeric_null_as_decimal(self):

        # The original SQLAlchemy test passes [None] as the expected
        # output value for the test, however the FlexODBC driver (or
        # possibly just DataFlex in general) won't allow NULL values
        # to be stored. We *could* force the conversion function in
        # the dialect's Decimal type-class to evaluate all 0.0000
        # decimals as None, but that would mean we couldn't store
        # 0-value decimals. The expected output for the test has been
        # modified accordingly.

        self._do_test(Numeric(precision=8, scale=4), [None], [PyDecimal("0.0")])

    @sa_testing.testing.requires.fetch_null_from_numeric
    def test_numeric_null_as_float(self):

        # The original SQLAlchemy test passes [None] as the expected
        # output value for the test, however the FlexODBC driver (or
        # possibly just DataFlex in general) won't allow NULL values
        # to be stored. We *could* force the conversion function in
        # the dialect's Decimal type-class to evaluate all 0.0 decimals
        # as None, but that would mean we couldn't store 0-value decimals.
        # The expected output for the test has been modified accordingly.

        self._do_test(
            Numeric(precision=8, scale=4, asdecimal=False),
            [None],
            [float("0.0")],
            filter_=lambda n: float(n) if n is not None else n,
        )

    @sa_testing.testing.requires.precision_numerics_general
    def test_precision_decimal(self):
        # DataFlex's numeric precision and scale max out at
        # 14 and 6 respectively, so trying to store or scalar
        # select anything larger is a fool's errand.
        numbers = {PyDecimal("54.234246"), PyDecimal("0.004354"), PyDecimal("900.0")}

        self._do_test(Numeric(precision=14, scale=6), numbers, numbers)

    def test_render_literal_float(self):
        self._literal_round_trip(
            Numeric(4),
            [15.7563, PyDecimal("15.7563")],
            [15.7563],
            filter_=lambda n: float(n) if (n is not None and round(n, 5) or None) else n,
        )

    @sa_testing.testing.emits_warning(r".*does \*not\* support Decimal objects natively")
    def test_render_literal_numeric(self):
        self._literal_round_trip(
            Numeric(precision=8, scale=4), [15.7563, PyDecimal("15.7563")], [PyDecimal("15.7563")],
        )

    @sa_testing.testing.emits_warning(r".*does \*not\* support Decimal objects natively")
    def test_render_literal_numeric_asfloat(self):
        self._literal_round_trip(
            Numeric(precision=8, scale=4, asdecimal=False),
            [15.7563, PyDecimal("15.7563")],
            [15.7563],
            filter_=lambda n: float(n) if n is not None else n,
        )


class OrderByLabelTest(_OrderByLabelTest):
    """Test the dialect's handling of `order by` {label} clauses."""

    @classmethod
    def define_tables(cls, metadata):
        """Create the table(s) required by the test(s)."""
        sa.Table(
            "some_table",
            metadata,
            sa.Column("id", Integer, primary_key=True, nullable=True),
            sa.Column("x", Integer),
            sa.Column("y", Integer),
            sa.Column("q", VarChar(50)),
            sa.Column("p", VarChar(50)),
        )

    @classmethod
    def insert_data(cls, connection):
        """Ensure that any data required by the test(s) exists in the relevant table(s)."""
        for param_set in [
            {"id": 1, "x": 1, "y": 2, "q": "q1", "p": "p3"},
            {"id": 2, "x": 2, "y": 3, "q": "q2", "p": "p2"},
            {"id": 3, "x": 3, "y": 4, "q": "q3", "p": "p1"},
        ]:
            connection.execute(cls.tables.some_table.insert(param_set))

    def _assert_result(self, query, expected):
        result = sa_testing.config.db.execute(query).fetchall()
        sa_testing.eq_(result, expected)

    def test_composed_int(self):
        table = self.tables.some_table

        clause = (table.c.x + table.c.y).label("lx")
        query = sa.select([clause]).order_by(clause)
        expected = [(3,), (5,), (7,)]

        self._assert_result(query, expected)

    def test_composed_int_desc(self):
        table = self.tables.some_table

        clause = (table.c.x + table.c.y).label("lx")
        query = sa.select([clause]).order_by(clause.desc())
        expected = [(7,), (5,), (3,)]

        self._assert_result(query, expected)

    def test_composed_multiple(self):
        table = self.tables.some_table

        lx = (table.c.x + table.c.y).label("lx")
        ly = (sa.func.lower(table.c.q) + table.c.p).label("ly")
        query = sa.select([lx, ly]).order_by(lx, ly.desc())
        expected = [(3, sa.util.u("q1p3")), (5, sa.util.u("q2p2")), (7, sa.util.u("q3p1"))]

        self._assert_result(query, expected)

    @sa_testing.testing.requires.group_by_complex_expression
    def test_group_by_composed(self):
        table = self.tables.some_table

        expr = (table.c.x + table.c.y).label("lx")
        query = sa.select([sa.func.count(table.c.id), expr]).group_by(expr).order_by(expr)
        expected = [(1, 3), (1, 5), (1, 7)]

        self._assert_result(query, expected)


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class PercentSchemaNamesTest(_PercentSchemaNamesTest):
    """Test the dialect's handling of percent schema names."""


# noinspection PyArgumentList
class QuotedNameArgumentTest(_QuotedNameArgumentTest):
    """Test the dialect's handling of quoted name arguments."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required by the test(s)."""

        sa.Table(
            "quote ' one",
            metadata,
            sa.Column("id", Integer),
            sa.Column("name", VarChar(50)),
            sa.Column("data", VarChar(50)),
            sa.Column("related_id", Integer),
            sa.PrimaryKeyConstraint("id", name="pk quote ' one"),
            sa.Index("ix quote ' one", "name"),
            sa.UniqueConstraint("data", name="uq quote' one",),
            sa.ForeignKeyConstraint(["id"], ["related.id"], name="fk quote ' one"),
            sa.CheckConstraint("name != 'foo'", name="ck quote ' one"),
            comment=r"""quote ' one comment""",
        )

        sa.Table(
            "related",
            metadata,
            sa.Column("id", Integer, primary_key=True),
            sa.Column("related", Integer),
            schema=None,
        )

    # noinspection PyMethodParameters
    def quote_fixtures(fn):
        """Quote a fixture."""
        return sa_testing.testing.combinations(
            ("quote ' one",), ('quote " two', sa_testing.testing.requires.symbol_names_w_double_quote),
        )(fn)

    @quote_fixtures
    def test_get_columns(self, name):
        insp = sa.inspect(sa_testing.testing.db)
        result = insp.get_columns(name)
        assert result

    @pytest.mark.skip(reason="Foreign Key reflection not supported by FlexODBC driver.")
    def test_get_foreign_keys(self, name):
        """Test the dialect's handling of foreign key reflection."""
        pass

    @pytest.mark.skip(reason="Index reflection not supported by FlexODBC driver.")
    def test_get_indexes(self, name):
        """Test the dialect's handling of index reflection."""
        pass

    @quote_fixtures
    def test_get_pk_constraint(self, name):
        insp = sa.inspect(sa_testing.testing.db)
        result = insp.get_pk_constraint(name)
        assert result

    @quote_fixtures
    def test_get_table_options(self, name):
        insp = sa.inspect(sa_testing.testing.db)
        result = insp.get_table_options(name)
        assert result is not None


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class ReturningTest(_ReturningTest):
    """Test the dialect's handling of returning clauses."""


class RowFetchTest(_RowFetchTest):
    """Test the dialect's handling of row fetches."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required by the test(s)."""
        sa.Table(
            "plain_pk",
            metadata,
            sa.Column("id", Integer, primary_key=True),
            sa.Column("data", VarChar(50)),
            schema=None,
        )
        sa.Table(
            "has_dates",
            metadata,
            sa.Column("id", Integer, primary_key=True),
            sa.Column("today", Timestamp),
            schema=None,
        )

    @classmethod
    def insert_data(cls, connection):
        """Ensure that any data required by the test(s) exists in the relevant table(s)."""
        for param_set in [
            {"id": 1, "data": "d1"},
            {"id": 2, "data": "d2"},
            {"id": 3, "data": "d3"},
        ]:
            connection.execute(cls.tables.plain_pk.insert(param_set))

        for param_set in [{"id": 1, "today": datetime(2006, 5, 12, 12, 0, 0)}]:
            connection.execute(cls.tables.has_dates.insert(param_set))

    def test_via_string(self):
        row = sa_testing.config.db.execute(self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)).first()

        sa_testing.eq_(row["id"], 1)
        sa_testing.eq_(row["data"], "d1")

    def test_via_int(self):
        row = sa_testing.config.db.execute(self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)).first()

        sa_testing.eq_(row[0], 1)
        sa_testing.eq_(row[1], "d1")

    def test_via_col_object(self):
        row = sa_testing.config.db.execute(self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)).first()

        sa_testing.eq_(row[self.tables.plain_pk.c.id], 1)
        sa_testing.eq_(row[self.tables.plain_pk.c.data], "d1")

    @pytest.mark.skip(reason="Scalar-SELECT-as-column unsupported by FlexODBC driver.")
    def test_row_w_scalar_select(self):
        """Test that a scalar select as a column is returned as such and that type conversion works.
        """
        date_table = self.tables.has_dates
        s = sa.select([date_table.c.today.label("x")]).as_scalar()
        s2 = sa.select([date_table.c.id, s.label("somelabel")])
        row = sa_testing.config.db.execute(s2).first()

        # This test wants to emit a query that looks something like:
        #
        # SELECT
        #   "has_dates"."id",
        #   (SELECT "x"."today" FROM "has_dates" AS "x") AS "somelabel"
        # FROM
        #   "has_dates"
        #
        # But FlexODBC can't handle anything even remotely close. It's
        # also unclear how similar functionality might be emulated.
        # Until that changes, this test will remain marked as `skip`.

        sa_testing.eq_(row["somelabel"], datetime(2006, 5, 12, 12, 0, 0))


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class SequenceCompilerTest(_SequenceCompilerTest):
    """Test the dialect's sequence compiler."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class SequenceTest(_SequenceTest):
    """Test the dialect's handling of sequences."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class ServerSideCursorsTest(_ServerSideCursorsTest):
    """Test the dialect's handling of server side cursors."""


class SimpleUpdateDeleteTest(_SimpleUpdateDeleteTest):
    """Test the dialect's handling of simple updates and deletes."""

    @classmethod
    def define_tables(cls, metadata):
        """Define the table(s) required by the test(s)."""
        sa.Table(
            "plain_pk",
            metadata,
            sa.Column("id", Integer, primary_key=True),
            sa.Column("data", VarChar(50)),
            schema=None,
        )

    @classmethod
    def insert_data(cls, connection):
        """Ensure that any data required by the test(s) exists in the relevant table(s)."""
        for param_set in [
            {"id": 1, "data": "d1"},
            {"id": 2, "data": "d2"},
            {"id": 3, "data": "d3"},
        ]:
            connection.execute(cls.tables.plain_pk.insert(param_set))

    def test_delete(self):
        table = self.tables.plain_pk

        query = table.delete().where(table.c.id == 2)
        result = sa_testing.config.db.execute(query)

        assert not result.is_insert
        assert not result.returns_rows

        check_query = table.select().order_by(table.c.id)
        check_result = sa_testing.config.db.execute(check_query).fetchall()

        expected = [(1, "d1"), (3, "d3")]

        assert check_result == expected

    def test_update(self):
        table = self.tables.plain_pk

        query = table.update().values({"data": "d2_new"}).where(table.c.id == 2)
        result = sa_testing.config.db.execute(query)

        assert not result.is_insert
        assert not result.returns_rows

        check_query = table.select().order_by(table.c.id)
        check_result = sa_testing.config.db.execute(check_query).fetchall()

        expected = [(1, "d1"), (2, "d2_new"), (3, "d3")]

        assert check_result == expected


class StringTest(_StringTest):
    """Test the dialect's handling of string data."""

    @sa_testing.testing.provide_metadata
    def _literal_round_trip(self, type_, input_, output, filter_=None):
        """Test literal rendering """
        df_literal_round_trip(self, type_, input_, output, filter_)

    @sa_testing.requirements.unbounded_varchar
    def test_nolength_string(self):
        metadata = sa.MetaData()
        foo = sa.Table("foo", metadata, sa.Column("one", VarChar))

        foo.create(sa_testing.config.db)
        foo.drop(sa_testing.config.db)

    def test_literal(self):
        # note that in Python 3, this invokes the Unicode
        # datatype for the literal part because all strings are unicode
        self._literal_round_trip(VarChar(40), ["some text"], ["some text"])

    @pytest.mark.skip(reason="Non-ASCII character storage emulation not currently implemented.")
    def test_literal_non_ascii(self):
        self._literal_round_trip(VarChar(40), [util.u("réve🐍 illé")], [sa.util.u("réve🐍 illé")])

    def test_literal_quoting(self):
        data = """some 'text' hey "hi there" that's text"""
        self._literal_round_trip(VarChar(40), [data], [data])

    def test_literal_backslashes(self):
        data = r"backslash one \ backslash two \\ end"
        self._literal_round_trip(VarChar(40), [data], [data])


@pytest.mark.skip(reason="FlexODBC driver does not support *any* DDL statements.")
class TableDDLTest(_TableDDLTest):
    """Test the dialect's handling of table DDL statements."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class TextTest(_TextTest):
    """Test the dialect's handling of text data."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class TimeMicrosecondsTest(_TimeMicrosecondsTest):
    """Test the dialect's handling of time-object microseconds."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class TimeTest(_TimeTest):
    """Test the dialect's handling of time data."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class TimestampMicrosecondsTest(_TimestampMicrosecondsTest):
    """Test the dialect's handling of timestamp data w/ microseconds."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class UnicodeTextTest(_UnicodeTextTest):
    """Test the dialect's handling of unicode text."""


@pytest.mark.skip(reason="Unsupported by FlexODBC driver.")
class UnicodeVarcharTest(_UnicodeVarcharTest):
    """Test the dialect's handling of unicode varchar data."""
