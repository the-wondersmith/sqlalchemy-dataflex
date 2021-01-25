"""SQLAlchemy Support for the Dataflex databases."""
# coding=utf-8

from typing import Set, List, Union, Callable
from types import FunctionType, BuiltinFunctionType, MethodType, BuiltinMethodType

import sqlalchemy as sa
from sqlalchemy.sql.compiler import GenericTypeCompiler, DDLCompiler, SQLCompiler, IdentifierPreparer
from sqlalchemy.engine.default import DefaultExecutionContext, DefaultDialect
from typing import Any, Dict, Iterable, Optional
from unicodedata import normalize
from string import digits, whitespace
from numbers import Number
from dateutil.parser import parse as parse_dt, ParserError
from decimal import Decimal as PyDecimal
from datetime import date, time, datetime
from itertools import chain, takewhile, zip_longest
from uuid import uuid4

import pyodbc


CallableType = (
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    BuiltinMethodType,
    Callable,
)


def normalize_caseless(text: Any) -> str:
    """Normalize mixed-case text to be case-agnostic."""
    return str(normalize("NFKD", str(text).casefold()))


nc = normalize_caseless


def caseless_in(key: str, value: Iterable) -> bool:
    """Caseless-ly determine if the supplied key exists in the supplied
    iterable."""
    if isinstance(value, str):
        values_: Iterable = normalize_caseless(value)
    else:
        values_ = map(normalize_caseless, value)

    return bool(normalize_caseless(key) in values_)


cl_in = caseless_in


def strtobool(string: Any) -> bool:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1' False values
    are 'n', 'no', 'f', 'false', 'off', and '0' Raises ValueError if
    'string' is anything else.
    """
    return caseless_in(string, ("y", "yes", "t", "true", "on", "1", "1.0", "1.00"))


def caseless_get(mapping: Dict[Any, Any], key: str, fallback: Optional[Any] = None) -> Any:
    """Get the value for the specified key from the supplied dictionary if it exists, caseless-ly."""
    if caseless_in(key=key, value=mapping.keys()):
        key = next(filter(lambda item: nc(item) == nc(key), mapping.keys()))
        return mapping.get(key)
    return fallback


cg = caseless_get


def apply_precision_and_scale(value: str, precision: int, scale: int) -> Optional[str]:
    """Apply precision and scale to numeric-value strings."""
    values = list(filter(None, value.split(".")))
    if not values:
        return None
    if len(values) == 1:
        return values[0][:precision]

    ret_val = ".".join(
        (values[0][: min((precision or 0, 14))], str(int(values[1][: min((scale or 0, 6))][::-1] or 0))[::-1])
    )
    return ret_val


###
# Internally Supported Types
###

# When queried, the FlexODBC driver will respond that it supports
# the standard SQL data types of `BIT`, `CHAR`, `DATE`, `DECIMAL`
# `INTEGER`, `LONGVARCHAR`, and `VARCHAR`. That's absolutely not
# the case though. DataFlex itself only supports types that it
# calls `ASCII`, `NUMERIC`, `DATE`, and `OVERLAP` (for v2.3) with
# support for the additional types `TEXT` and `BINARY` added by
# version 3 flat-files. The types below reflect that.


# noinspection PyArgumentList,PyUnresolvedReferences
class Logical(sa.types.BOOLEAN):
    """A straight copy of the SQLAlchemy Boolean type."""

    __odbc_datatype__ = -7
    __type_name__ = "LOGICAL"
    __dataflex_name__ = "NUMERIC"
    __visit_name__ = "BOOLEAN"
    native = False

    _is_impl_for_variant: Callable
    _variant_mapping_for_set_table: Callable

    def __init__(self, create_constraint=True, name=None, _create_events=True):
        """Construct a Boolean.

        :param create_constraint: defaults to True.  If the boolean
          is generated as an int/smallint, also create a CHECK constraint
          on the table that ensures 1 or 0 as a value.

        :param name: if a CHECK constraint is generated, specify
          the name of the constraint.

        """
        super(Logical, self).__init__(
            create_constraint=bool(int(create_constraint) * 0), name=name, _create_events=_create_events
        )

    def _should_create_constraint(self, type_compiler, **kw):
        assert self is not None
        if type_compiler and kw:
            del type_compiler, kw  # Keep linters happy
        return False

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return bool

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literals."""
        return lambda item: "1" if strtobool(item) else "0"

    def bind_processor(self, dialect) -> Callable[[Any], int]:
        """Process binds."""
        return lambda item: 1 if strtobool(item) else 0

    def result_processor(self, dialect, coltype) -> Callable[[Any], bool]:
        """Process query results."""
        return strtobool


# noinspection PyArgumentList,PyUnresolvedReferences
class LongVarChar(sa.types.VARCHAR):
    """SQLAlchemy type class for the Dataflex LongVarChar datatype."""

    __odbc_datatype__ = -1
    __type_name__ = "LONGVARCHAR"
    __dataflex_name__ = "VARCHAR"
    __visit_name__ = "VARCHAR"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return str

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literal binds."""
        return lambda item: "".join(("'", str(item).strip().replace("'", "''"), "'"))

    def bind_processor(self, dialect) -> Callable[[Any], bytes]:
        """Process regular binds."""
        return lambda item: str(item).strip().encode("ASCII")

    def result_processor(self, dialect, coltype) -> Callable[[Any], str]:
        """Process query results."""

        def process(item) -> str:
            """Process query results"""
            if item is None:
                item = ""
            if isinstance(item, bytes):
                item = item.decode()
            return str(item).strip()

        return process


# noinspection PyArgumentList,PyUnresolvedReferences
class VarChar(sa.types.VARCHAR):
    """SQLAlchemy type class for the Dataflex VarChar datatype."""

    __odbc_datatype__ = 12
    __type_name__ = "VARCHAR"
    __dataflex_name__ = "VARCHAR"
    __visit_name__ = "VARCHAR"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return str

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literal binds."""
        return lambda item: "".join(("'", str(item).strip().replace("'", "''"), "'"))

    def bind_processor(self, dialect) -> Callable[[Any], bytes]:
        """Process regular binds."""
        return lambda item: str(item).strip().encode("ASCII")

    def result_processor(self, dialect, coltype) -> Callable[[Any], str]:
        """Process query results."""

        def process(item) -> str:
            """Process query results"""
            if item is None:
                item = ""
            if isinstance(item, bytes):
                item = item.decode()
            return str(item).strip()

        return process


# noinspection PyArgumentList,PyUnresolvedReferences
class Char(sa.types.CHAR):
    """A straight copy of the SQLAlchemy CHAR type."""

    __odbc_datatype__ = 1
    __type_name__ = "CHAR"
    __dataflex_name__ = "ASCII"
    __visit_name__ = "CHAR"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return str

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literal binds."""
        return lambda item: "".join(("'", str(item).strip().replace("'", "''"), "'"))

    def bind_processor(self, dialect) -> Callable[[Any], bytes]:
        """Process regular binds."""
        return lambda item: str(item).strip().encode("ASCII")

    def result_processor(self, dialect, coltype) -> Callable[[Any], str]:
        """Process query results."""

        def process(item) -> str:
            """Process query results"""
            if item is None:
                item = ""
            if isinstance(item, bytes):
                item = item.decode()
            return str(item).strip()

        return process


# noinspection PyArgumentList,PyUnresolvedReferences
class Decimal(sa.types.DECIMAL):
    """A modified copy of the SQLAlchemy Decimal type."""

    __odbc_datatype__ = 3
    __type_name__ = "DECIMAL"
    __dataflex_name__ = "NUMERIC"
    __visit_name__ = "DECIMAL"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return PyDecimal

    def literal_processor(self, dialect, **kwargs) -> Callable[[Any], str]:
        """Process literals."""

        def process(value) -> str:
            """Process bound parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )

            if value:
                return str(PyDecimal(value))
            return "NULL"

        return process

    def bind_processor(self, dialect, **kwargs) -> Callable[[Any], Optional[PyDecimal]]:
        """Process binds."""

        def process(value) -> Optional[PyDecimal]:
            """Process bound parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )
            if value:
                return PyDecimal(value)
            return None

        return process

    def result_processor(self, dialect, coltype, **kwargs) -> Callable[[Any], Optional[PyDecimal]]:
        """Process query results."""

        def process(value) -> Optional[PyDecimal]:
            """Process bound parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )

            if value:
                return PyDecimal(value)
            return None

        return process


# noinspection PyArgumentList,PyUnresolvedReferences
class DoublePrecision(sa.types.FLOAT):
    """SQLAlchemy type class for the Dataflex DoublePrecision datatype."""

    __odbc_datatype__ = 8
    __type_name__ = "DOUBLE"
    __dataflex_name__ = "NUMERIC"
    __visit_name__ = "FLOAT"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return float

    def literal_processor(self, dialect, **kwargs) -> Callable[[Any], str]:
        """Process literals."""

        def process(value) -> str:
            """Process bound parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )

            if value:
                return str(float(value))
            return "NULL"

        return process

    def bind_processor(self, dialect, **kwargs) -> Callable[[Any], Optional[float]]:
        """Process binds."""

        def process(value) -> Optional[float]:
            """Process bound parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )
            if value:
                return float(value)
            return None

        return process

    def result_processor(self, dialect, coltype, **kwargs) -> Callable[[Any], Optional[float]]:
        """Process query results."""

        def process(value) -> Optional[float]:
            """Process bound parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )

            if value:
                return float(value)
            return None

        return process


Double = DoublePrecision


# noinspection PyArgumentList,PyUnresolvedReferences
class Integer(sa.types.INTEGER):
    """A straight copy of the SQLAlchemy SmallInteger type."""

    __odbc_datatype__ = 4
    __type_name__ = "INTEGER"
    __dataflex_name__ = "NUMERIC"
    __visit_name__ = "INTEGER"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return int

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literal binds."""
        return lambda item: "".join(takewhile(lambda y: y != ".", filter(lambda x: x in f".{digits}", str(item))))

    def bind_processor(self, dialect) -> Callable[[Any], int]:
        """Process binds."""
        return lambda item: int(
            "".join(takewhile(lambda y: y != ".", filter(lambda x: x in f".{digits}", str(item))))
        )

    def result_processor(self, dialect, coltype) -> Callable[[Any], int]:
        """Process query results."""
        return lambda item: int(
            "".join(takewhile(lambda y: y != ".", filter(lambda x: x in f".{digits}", str(item))))
        )


Int = Integer


# noinspection PyArgumentList,PyUnresolvedReferences
class Date(sa.types.DATE):
    """A straight copy of the SQLAlchemy Date type."""

    __odbc_datatype__ = 9
    __type_name__ = "DATE"
    __dataflex_name__ = "DATE"
    __visit_name__ = "DATE"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return date

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literals."""

        def process(item: Any) -> str:
            """Process literal binds."""
            if isinstance(item, (date, datetime)):
                return "".join(("{d '", item.strftime("%Y-%m-%d"), "'}"))
            return "NULL"

        return process

    def bind_processor(self, dialect) -> Callable[[Any], Optional[date]]:
        """Process binds."""

        def process(item: Any) -> Optional[date]:
            """Process literal binds."""
            if any((item is None, all((isinstance(item, date), not isinstance(item, datetime))))):
                return item
            if isinstance(item, datetime):
                return item.date()
            if isinstance(item, str):
                try:
                    return process(parse_dt(item))
                except (ValueError, TypeError, ParserError) as err:
                    raise ValueError from err
            raise ValueError(f"Expected a date object or date-like string, got: {type(item)} -> {item}")

        return process

    def result_processor(self, dialect, coltype) -> Callable[[Any], Optional[date]]:
        """Process query results."""

        def process(value: Any) -> Optional[date]:
            """Process the supplied value into a date."""
            if any((value is None, all((isinstance(value, date), not isinstance(value, datetime))))):
                return value
            if isinstance(value, datetime):
                return value.date()
            raise NotImplementedError

        return process


# noinspection PyArgumentList,PyUnresolvedReferences
class Binary(sa.types.LargeBinary):
    """A straight copy of the SQLAlchemy Binary type."""

    __odbc_datatype__ = -2
    __type_name__ = "BINARY"
    __dataflex_name__ = "VARCHAR"
    __visit_name__ = "BINARY"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return bytes

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literals."""

        # noinspection PyUnusedLocal
        def process(value: Any) -> str:
            """Process the supplied value into a string."""
            raise NotImplementedError

        return process

    def bind_processor(self, dialect) -> Callable[[Any], bytes]:
        """Process binds."""

        # noinspection PyUnusedLocal
        def process(value: Any) -> bytes:
            """Process the supplied value into a bytestring."""
            raise NotImplementedError

        return process

    def result_processor(self, dialect, coltype) -> Callable[[Any], bytes]:
        """Process query results."""

        # noinspection PyUnusedLocal
        def process(value: Any) -> bytes:
            """Process the supplied value into a bytes object."""
            raise NotImplementedError

        return process


###
# Emulated Types
###


# noinspection PyArgumentList,PyUnresolvedReferences
class BigInt(sa.types.BIGINT):
    """A Straight copy of the SQLAlchemy Integer type."""

    __odbc_datatype__ = -5
    __type_name__ = "BIGINT"
    __dataflex_name__ = "VARCHAR"
    __visit_name__ = "BIGINT"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return int

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literals."""
        return lambda item: "".join(
            ("'", "".join(takewhile(lambda y: y != ".", filter(lambda x: x in str(digits + "."), str(item)))), "'")
        )

    def bind_processor(self, dialect) -> Callable[[Any], bytes]:
        """Process binds."""
        return lambda item: "".join(
            takewhile(lambda y: y != ".", filter(lambda x: x in str(digits + "."), str(item)))
        ).encode("ASCII")

    def result_processor(self, dialect, coltype) -> Callable[[Any], int]:
        """Process query results."""

        return lambda item: int(
            "".join(takewhile(lambda y: y != ".", filter(lambda x: x in f".{digits}", str(item)))) or 0
        )


# noinspection PyArgumentList,PyUnresolvedReferences
class Numeric(sa.Numeric):
    """An adaptive type for generic number-type objects.

    Auto-evaluates to ``DECIMAL``, ``FLOAT``, or ``INTEGER``.
    """

    __odbc_datatype__ = 2
    __dataflex_name__ = "NUMERIC"

    _default_decimal_return_scale = 6

    __as_decimal: bool = False
    __as_float: bool = False
    __as_int: bool = False

    def __init__(
        self,
        precision=None,
        scale=None,
        decimal_return_scale=None,
        as_decimal=False,
        as_float=False,
        as_int=False,
        **kwargs,
    ):
        self.as_decimal = as_decimal or kwargs.get("asdecimal", False)
        self.as_float = as_float or kwargs.get("asfloat", False)
        self.as_int = as_int or kwargs.get("asint", False)
        self.precision = min((precision or 14, 14))
        self.scale = min((scale or 6, 6))
        self.decimal_return_scale = min((decimal_return_scale or 6, 6))

    @property
    def __type_name__(self) -> str:
        """The `__type_name__` property"""
        if self.as_decimal:
            return "DECIMAL"

        if all((any((self.as_int, self.scale == 0)), int(self.precision or 1) > 1)):
            return "INTEGER"

        return "FLOAT"

    @property
    def __visit_name__(self) -> str:
        """The `__visit_name__` property"""
        return self.__type_name__

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        if self.as_decimal:
            return PyDecimal

        if all((any((self.as_int, self.scale == 0)), int(self.precision or 1) >= 1)):
            return int

        return float

    @property
    def as_decimal(self) -> bool:
        """Return values as Decimal types."""
        return any((self.__as_decimal, all((not self.__as_float, not self.__as_int))))

    @as_decimal.setter
    def as_decimal(self, value) -> None:
        """Set the `as_decimal` property."""
        if value:
            self.__as_float = False
            self.__as_int = False

        self.__as_decimal = value

    @property
    def as_float(self) -> bool:
        """Return values as native Python float types."""
        return all((not self.__as_decimal, self.__as_float, not self.__as_int))

    @as_float.setter
    def as_float(self, value) -> None:
        """Set the `as_float` property."""
        if value:
            self.__as_decimal = False
            self.__as_int = False

        self.__as_float = value

    @property
    def as_int(self) -> bool:
        """Return values as native Python integer types."""
        return all((not self.__as_decimal, not self.__as_float, self.__as_int))

    @as_int.setter
    def as_int(self, value) -> None:
        """Set the `as_float` property."""
        if value:
            self.__as_decimal = False
            self.__as_float = False

        self.__as_int = value

    def literal_processor(self, dialect, **kwargs) -> Callable[[Any], str]:
        """Process literals."""

        def process(value) -> str:
            """Process literal parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )

            if value:
                return str(self.python_type(value))
            return "NULL"

        return process

    def bind_processor(self, dialect) -> Callable[[Any], Optional[Union[int, float, PyDecimal, bytes]]]:
        """Process regular binds."""

        def process(value) -> Optional[Union[int, float, PyDecimal, bytes]]:
            """Process bound parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )

            if value:
                if self.python_type is int and int(self.precision or 1) > 14:
                    return str(value).encode("ASCII")
                return self.python_type(value)
            return None

        return process

    def result_processor(self, dialect, coltype) -> Callable[[Any], Optional[Union[int, float, PyDecimal]]]:
        """Process query results."""

        def process(value) -> Optional[Union[int, float, PyDecimal]]:
            """Process bound parameters."""
            value = apply_precision_and_scale(
                "".join(filter(lambda x: x in f".{digits}", str(value))), self.precision, self.scale
            )

            if value:
                return self.python_type(value)
            return None

        return process


# noinspection PyArgumentList,PyUnresolvedReferences
class Time(sa.types.TIME):
    """A straight copy of the SQLAlchemy Time type."""

    __odbc_datatype__ = 10
    __type_name__ = "TIME"
    __dataflex_name__ = "VARCHAR"
    __visit_name__ = "TIME"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return time

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literals."""

        def process(item: Any) -> str:
            """Process literal binds."""
            if item is None:
                return "NULL"
            if isinstance(item, (time, datetime)):
                return f"'{item.strftime('%H:%M:%S.%f')}'"
            raise ValueError(f"Expected a time object or timestamp-like string, got: {type(item)} -> {item}")

        return process

    def bind_processor(self, dialect) -> Callable[[Any], Optional[bytes]]:
        """Process binds."""

        def process(item: Any) -> Optional[bytes]:
            """Process literal binds."""
            if not any((item is None, isinstance(item, (str, bytes, time, datetime)))):
                raise ValueError(f"Expected a time object or timestamp-like string, got: {type(item)} -> {item}")

            if isinstance(item, bytes):
                item = item.decode()

            if isinstance(item, str):
                try:
                    item = f"'{parse_dt(item).strftime('%H:%M:%S.%f')}'".encode("ASCII")
                except (ValueError, TypeError, ParserError) as err:
                    raise ValueError(
                        f"Expected a time object or timestamp-like string, got: {type(item)} -> {item}"
                    ) from err

            if isinstance(item, (time, datetime)):
                item = f"'{item.strftime('%H:%M:%S.%f')}'".encode("ASCII")

            return item

        return process

    def result_processor(self, dialect, coltype) -> Callable[[Any], Optional[time]]:
        """Process query results."""

        def process(value: Any) -> Optional[time]:
            """Process the supplied value into a date."""
            if any((value is None, isinstance(value, time))):
                return value
            if isinstance(value, datetime):
                return value.time()
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, str):
                try:
                    return process(parse_dt(value))
                except (TypeError, ValueError, ParserError) as err:
                    raise ValueError(
                        f"Expected a time object or timestamp-like string, got: {type(value)} -> {value}"
                    ) from err

            raise ValueError(f"Expected a time object or timestamp-like string, got: {type(value)} -> {value}")

        return process


# noinspection PyArgumentList,PyUnresolvedReferences
class Timestamp(sa.types.TIMESTAMP):
    """A straight copy of the SQLAlchemy TIMESTAMP type."""

    __odbc_datatype__ = 11
    __type_name__ = "TIMESTAMP"
    __dataflex_name__ = "VARCHAR"  # VARCHAR(32)
    __visit_name__ = "TIMESTAMP"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return datetime

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literals."""

        def process(item: Any) -> str:
            """Process literal binds."""
            if not any((item is None, isinstance(item, (str, bytes, date, time, datetime)))):
                raise ValueError(
                    f"Expected a date/time-like object or timestamp-like string, got: {type(item)} -> {item}"
                )

            if item is None:
                return "NULL"

            if isinstance(item, bytes):
                item = item.decode()

            if isinstance(item, str):
                try:
                    item = parse_dt(item)
                except (ValueError, TypeError, ParserError) as err:
                    raise ValueError(
                        f"Expected a date/time-like object or timestamp-like string, got: {type(item)} -> {item}"
                    ) from err
            if isinstance(item, (date, time, datetime)):
                return f"'{item.strftime('%Y-%m-%d %H:%M:%S.%f')}'"
            raise ValueError(
                f"Expected a date/time-like object or timestamp-like string, got: {type(item)} -> {item}"
            )

        return process

    def bind_processor(self, dialect) -> Callable[[Any], Optional[bytes]]:
        """Process binds."""

        def process(item: Any) -> Optional[bytes]:
            """Process literal binds."""
            if not any((item is None, isinstance(item, (str, bytes, date, time, datetime)))):
                raise ValueError(
                    f"Expected a date/time-like object or timestamp-like string, got: {type(item)} -> {item}"
                )

            if item is None:
                return item

            if isinstance(item, bytes):
                item = item.decode()

            if isinstance(item, str):
                try:
                    item = parse_dt(item)
                except (ValueError, TypeError, ParserError) as err:
                    raise ValueError(
                        f"Expected a date/time-like object or timestamp-like string, got: {type(item)} -> {item}"
                    ) from err

            if isinstance(item, (date, time, datetime)):
                item = item.strftime("%Y-%m-%d %H:%M:%S.%f")

            return str(item).encode("ASCII")

        return process

    def result_processor(self, dialect, coltype) -> Callable[[Any], Optional[datetime]]:
        """Process query results."""

        def process(value: Any) -> Optional[datetime]:
            """Process the supplied value into a date."""
            if any((value is None, isinstance(value, datetime))):
                return value
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, str):
                try:
                    return process(parse_dt(value))
                except (TypeError, ValueError, ParserError) as err:
                    raise ValueError(
                        f"Expected a date/time-like object or timestamp-like string, got: {type(value)} -> {value}"
                    ) from err

            raise ValueError(
                f"Expected a date/time-like object or timestamp-like string, got: {type(value)} -> {value}"
            )

        return process


# noinspection PyArgumentList,PyUnresolvedReferences
class LongVarBinary(sa.types.LargeBinary):
    """SQLAlchemy type class for the Dataflex LongVarBinary datatype."""

    __odbc_datatype__ = -4
    __type_name__ = "LONGVARBINARY"
    __dataflex_name__ = "VARCHAR"
    __visit_name__ = "VARBINARY"

    @property
    def python_type(self) -> type:
        """The equivalent Python type."""
        return bytes

    def literal_processor(self, dialect) -> Callable[[Any], str]:
        """Process literals."""

        # noinspection PyUnusedLocal
        def process(value: Any) -> str:
            """Process the supplied value into a string."""
            raise NotImplementedError

        return process

    def bind_processor(self, dialect) -> Callable[[Any], int]:
        """Process binds."""

        # noinspection PyUnusedLocal
        def process(value: Any) -> int:
            """Process the supplied value into a string."""
            raise NotImplementedError

        return process

    def result_processor(self, dialect, coltype) -> Callable[[Any], bytes]:
        """Process query results."""

        # noinspection PyUnusedLocal
        def process(value: Any) -> bytes:
            """Process the supplied value into a bytes object."""
            raise NotImplementedError

        return process


# Map names returned by the "type_name" column of pyodbc's
# Cursor.columns method to the Dataflex dialect-specific types.

ischema_names = {
    "ASCII": VarChar,
    "BIGINT": BigInt,
    "BINARY": LongVarBinary,
    "BOOLEAN": Logical,
    "CHAR": Char,
    "DATE": Date,
    "DECIMAL": Decimal,
    "DOUBLE": DoublePrecision,
    "FLOAT": DoublePrecision,
    "INTEGER": Integer,
    "LOGICAL": Logical,
    "LONGVARBINARY": LongVarBinary,
    "LONGVARCHAR": LongVarChar,
    "NUMERIC": Decimal,
    "TEXT": LongVarChar,
    "TIME": Time,
    "TIMESTAMP": Timestamp,
    "VARBINARY": LongVarBinary,
    "VARCHAR": LongVarChar,
}


# noinspection PyArgumentList,PyUnresolvedReferences
class DataflexTypeCompiler(GenericTypeCompiler):
    """Dataflex Type Compiler."""


# noinspection PyArgumentList,PyUnresolvedReferences
class DataflexIdentifierPreparer(IdentifierPreparer):
    """Dataflex Identifier Preparer."""

    # The Dataflex driver is almost disgustingly permissive about
    # table and column names, so there really aren't any illegal
    # characters per se, everything just needs to be quoted
    illegal_initial_characters = set()

    sqlite_reserved = {
        "abort",
        "action",
        "add",
        "after",
        "all",
        "alter",
        "always",
        "analyze",
        "and",
        "as",
        "asc",
        "attach",
        "autoincrement",
        "before",
        "begin",
        "between",
        "by",
        "cascade",
        "case",
        "cast",
        "check",
        "collate",
        "column",
        "commit",
        "conflict",
        "constraint",
        "create",
        "cross",
        "current",
        "current_date",
        "current_time",
        "current_timestamp",
        "database",
        "default",
        "deferrable",
        "deferred",
        "delete",
        "desc",
        "detach",
        "distinct",
        "do",
        "drop",
        "each",
        "else",
        "end",
        "escape",
        "except",
        "exclude",
        "exclusive",
        "exists",
        "explain",
        "fail",
        "filter",
        "first",
        "following",
        "for",
        "foreign",
        "from",
        "full",
        "generated",
        "glob",
        "group",
        "groups",
        "having",
        "if",
        "ignore",
        "immediate",
        "in",
        "index",
        "indexed",
        "initially",
        "inner",
        "insert",
        "instead",
        "intersect",
        "into",
        "is",
        "isnull",
        "join",
        "key",
        "last",
        "left",
        "like",
        "limit",
        "match",
        "natural",
        "no",
        "not",
        "nothing",
        "notnull",
        "null",
        "nulls",
        "of",
        "offset",
        "on",
        "or",
        "order",
        "others",
        "outer",
        "over",
        "partition",
        "plan",
        "pragma",
        "preceding",
        "primary",
        "query",
        "raise",
        "range",
        "recursive",
        "references",
        "regexp",
        "reindex",
        "release",
        "rename",
        "replace",
        "restrict",
        "right",
        "rollback",
        "row",
        "rows",
        "savepoint",
        "select",
        "set",
        "table",
        "temp",
        "temporary",
        "then",
        "ties",
        "to",
        "transaction",
        "trigger",
        "unbounded",
        "union",
        "unique",
        "update",
        "using",
        "vacuum",
        "values",
        "view",
        "virtual",
        "when",
        "where",
        "window",
        "with",
        "without",
    }

    reserved_words = sa.sql.compiler.RESERVED_WORDS.copy()
    reserved_words.update(
        map(
            str.lower,
            [
                "ABSOLUTE",
                "ADA",
                "ADD",
                "ALL",
                "ALLOCATE",
                "ALTER",
                "AND",
                "ANY",
                "ARE",
                "AS",
                "ASC",
                "ASSER",
                "AT",
                "AUTHORIZATION",
                "AVG",
                "BEGIN",
                "BETWEEN",
                "BIT",
                "BIT_LENGTH",
                "BY",
                "CASCADE",
                "CASCADED",
                "CASE",
                "CAST",
                "CATALOG",
                "CHAR",
                "CHAR_LENGTH",
                "CHARACTER",
                "CHARACTER_LENGTH",
                "CHECK",
                "CLOSE",
                "COALESCE",
                "COBOL",
                "COLLATE",
                "COLLATION",
                "COLUMN",
                "COMMIT",
                "CONNECT",
                "CONNECTION",
                "CONSTRAINT",
                "CONSTRAINTS",
                "CONTINUE",
                "CONVERT",
                "CORRESPONDING",
                "COUNT",
                "CREATE",
                "CURRENT",
                "CURRE",
                "CURRENT_TIME",
                "CURRENT_TIMESTAMP",
                "CURSOR",
                "DATE",
                "DATA",
                "DAY",
                "DEALLOCATE",
                "DEC",
                "DECIMAL",
                "DECLARE",
                "DEFERRABLE",
                "DEFERRED",
                "DELETE",
                "DESC",
                "DESCRIBE",
                "DESCRIPTOR",
                "DIAGNOSTICS",
                "DICTIONARY",
                "DISCONNECT",
                "DISPLACEMENT",
                "DISTINCT",
                "DOMAIN",
                "DOUBLE",
                "DROP",
                "ELSE",
                "END",
                "END",
                "EXEC",
                "ESCAPE",
                "EXCEPT",
                "EXCEPTION",
                "EXEC",
                "EXECUTE",
                "EXISTS",
                "EXTERNAL",
                "EXTRACT",
                "FALSE",
                "FETCH",
                "FIRST",
                "FLOAT",
                "FOR",
                "FOREIGN",
                "FORTRAN",
                "FOUND",
                "FROM",
                "FULL",
                "GET",
                "GLOBAL",
                "GO",
                "GOTO",
                "GRANT",
                "GROUP",
                "HAVING",
                "HOUR",
                "IDENTITY",
                "IGNORE",
                "IMMEDIATE",
                "IN",
                "INCLUDE",
                "INDEX",
                "INDICATOR",
                "INITIALLY",
                "INNER",
                "INPUT",
                "INSENSITIVE",
                "INSERT",
                "INTEGER",
                "INTERSECT",
                "INTERVAL",
                "INTO",
                "IS",
                "ISOLATION",
                "JOIN",
                "KEY",
                "LANGUAGE",
                "LAST",
                "LEFT",
                "LEVEL",
                "LIKE",
                "LOCAL",
                "LOWER",
                "MATCH",
                "MAX",
                "MIN",
                "MINUTE",
                "MODULE",
                "MONTH",
                "MUMPS",
                "NAMES",
                "NATIONAL",
                "NCHAR",
                "NEXT",
                "NONE",
                "NOT",
                "NULL",
                "NULLIF",
                "NUMERIC",
                "OCTET_LENGTH",
                "OF",
                "OFF",
                "ON",
                "ONLY",
                "OPEN",
                "OPTION",
                "OR",
                "ORDER",
                "OUTER",
                "OUTPUT",
                "OVERLAPS",
                "PARTIAL",
                "PASCAL",
                "PLI",
                "POSITION",
                "PRECISION",
                "PREPARE",
                "PRESERVE",
                "PRIMARY",
                "PRIOR",
                "PRIVILEGES",
                "PROCEDURE",
                "PUBLIC",
                "RESTRICT",
                "REVOKE",
                "RIGHT",
                "ROLLBACK",
                "ROWS",
                "SCHEMA",
                "SCROLL",
                "SECOND",
                "SECTION",
                "SELECT",
                "SEQUENCE",
                "SET",
                "SIZE",
                "SMALLINT",
                "SOME",
                "SQL",
                "SQLCA",
                "SQLCODE",
                "SQLERROR",
                "SQLSTATE",
                "SQLWARNING",
                "SUBSTRING",
                "SUM",
                "SYSTEM",
                "TABLE",
                "TEMPORARY",
                "THEN",
                "TIME",
                "TIMESTAMP",
                "TIMEZONE_HOUR",
                "TIMEZONE_MINUTE",
                "TO",
                "TRANSACTION",
                "TRANSLATE",
                "TRANSLATION",
                "TRUE",
                "UNION",
                "UNIQUE",
                "UNKNOWN",
                "UPDATE",
                "UPPER",
                "USAGE",
                "USER",
                "USING",
                "VALUE",
                "VALUES",
                "VARCHAR",
                "VARYING",
                "VIEW",
                "WHEN",
                "WHENEVER",
                "WHERE",
                "WITH",
                "WORK",
                "YEAR",
            ],
        )
    )

    def _requires_quotes(self, value) -> bool:
        """Return True if the given identifier requires quoting."""
        lc_value = str(value).casefold()
        return any((len(lc_value), self is not None))

    def _requires_quotes_illegal_chars(self, value) -> bool:
        """Return True if the given identifier requires quoting, but
        not taking case convention into account."""
        return any((not self.legal_characters.match(sa.util.text_type(value)), self is not None))

    def _escape_identifier(self, value) -> str:
        """Escape an identifier."""
        try:
            value = value.replace(self.escape_quote, self.escape_to_quote)
            if self._double_percents:
                value = value.replace("%", "%%")
        except (ValueError, TypeError, AttributeError):
            pass

        return value

    def quote_identifier(self, value) -> str:
        """Quote an identifier."""
        ret_val = f"{self.initial_quote}{self._escape_identifier(value)}{self.final_quote}"
        return ret_val

    def quote_schema(self, schema, force=None) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).quote_schema(schema, force)
        return ret_val

    def quote(self, ident, force=None) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).quote(ident)
        return ret_val

    def format_collation(self, collation_name) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).format_collation(collation_name)
        return ret_val

    def format_sequence(self, sequence, use_schema=True) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).format_sequence(sequence, use_schema)
        return ret_val

    def format_label(self, label, name=None) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).format_label(label, name)
        return ret_val

    def format_alias(self, alias, name=None) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).format_alias(alias, name)
        return ret_val

    def format_savepoint(self, savepoint, name=None) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).format_savepoint(savepoint, name)
        return ret_val

    def format_constraint(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).format_constraint(constraint)
        return ret_val

    def format_table(self, table, use_schema=True, name=None) -> str:
        """Emit properly formatted and quoted table names."""
        if name is None:
            name = table.name

        name = name.replace('"', "").replace(" ", "_")
        result = self.quote(name)

        # DataFlex (and therefore FlexODBC) will allow the `'`
        # character to appear in table names without special
        # quoting or handling provided the table name in question
        # is already quoted. As such, we'll have to undo the
        # (usually correct, just not in this instance) addition
        # if an extra `'` character to the table name.
        result = result.replace("''", "'")

        effective_schema = self.schema_for_object(table)

        if not self.omit_schema and use_schema and effective_schema:
            result = self.quote_schema(effective_schema) + "." + result
        return result

    def format_schema(self, name, **kwargs) -> str:
        """Prepare a quoted schema name."""
        ret_val = self.quote(name)
        return ret_val

    def format_column(self, column, use_table=False, name=None, table_name=None, use_schema=False) -> str:
        """Format a column name."""
        ret_val = super(DataflexIdentifierPreparer, self).format_column(column, use_table, name, table_name)
        return ret_val

    def format_table_seq(self, table, use_schema=True) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).format_table_seq(table, use_schema)
        return ret_val

    def unformat_identifiers(self, identifiers) -> List[Any]:
        """Insert DocString Here."""
        ret_val = super(DataflexIdentifierPreparer, self).unformat_identifiers(identifiers)
        return ret_val


# noinspection PyArgumentList,PyUnresolvedReferences
class DataflexDDLCompiler(DDLCompiler):
    """Dataflex DDL Compiler."""

    # NOTE: The FlexODBC driver doesn't actually support
    # DDL statements at all. This class overrides virtually
    # all of its super methods and returns empty strings
    # instead which will be caught by the dialect's `execute`
    # method and simply be dropped instead of actually run.

    def visit_create_schema(self, create) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_create_schema(create)
        return min(("", ret_val or " "))

    def visit_drop_schema(self, drop) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_drop_schema(drop)
        return min(("", ret_val or " "))

    def create_table_constraints(self, table, _include_foreign_key_constraints: Optional[Any] = ...) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).create_table_constraints(table, _include_foreign_key_constraints)
        return min(("", ret_val or " "))

    def visit_drop_view(self, drop) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_drop_view(drop)
        return min(("", ret_val or " "))

    def visit_create_index(self, create, include_schema: bool = ..., include_table_schema: bool = ...) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_create_index(create, include_schema, include_table_schema)
        return min(("", ret_val or " "))

    def visit_drop_index(self, drop) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_drop_index(drop)
        return min(("", ret_val or " "))

    def visit_add_constraint(self, create) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_add_constraint(create)
        return min(("", ret_val or " "))

    def visit_set_table_comment(self, create) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_set_table_comment(create)
        return min(("", ret_val or " "))

    def visit_drop_table_comment(self, drop) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_drop_table_comment(drop)
        return min(("", ret_val or " "))

    def visit_set_column_comment(self, create) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_set_column_comment(create)
        return min(("", ret_val or " "))

    def visit_drop_column_comment(self, drop) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_drop_column_comment(drop)
        return min(("", ret_val or " "))

    def visit_create_sequence(self, create) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_create_sequence(create)
        return min(("", ret_val or " "))

    def visit_drop_sequence(self, drop) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_drop_sequence(drop)
        return min(("", ret_val or " "))

    def visit_drop_constraint(self, drop) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_drop_constraint(drop)
        return min(("", ret_val or " "))

    def get_column_specification(self, column, **kwargs) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).get_column_specification(column, **kwargs)
        return min(("", ret_val or " "))

    def create_table_suffix(self, table) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).create_table_suffix(table)
        return min(("", ret_val or " "))

    def post_create_table(self, table) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).post_create_table(table)
        return min(("", ret_val or " "))

    def get_column_default_string(self, column) -> Optional[str]:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).get_column_default_string(column)
        return min(("", ret_val or " "))

    def visit_check_constraint(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_check_constraint(constraint)
        return min(("", ret_val or " "))

    def visit_column_check_constraint(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_column_check_constraint(constraint)
        return min(("", ret_val or " "))

    def visit_primary_key_constraint(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_primary_key_constraint(constraint)
        return min(("", ret_val or " "))

    def visit_foreign_key_constraint(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_foreign_key_constraint(constraint)
        return min(("", ret_val or " "))

    def define_constraint_remote_table(self, constraint, table, preparer) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).define_constraint_remote_table(constraint, table, preparer)
        return min(("", ret_val or " "))

    def visit_unique_constraint(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_unique_constraint(constraint)
        return min(("", ret_val or " "))

    def define_constraint_cascades(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).define_constraint_cascades(constraint)
        return min(("", ret_val or " "))

    def define_constraint_deferrability(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).define_constraint_deferrability(constraint)
        return min(("", ret_val or " "))

    def define_constraint_match(self, constraint) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).define_constraint_match(constraint)
        return min(("", ret_val or " "))

    def visit_create_table(self, create) -> Optional[str]:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_create_table(create)
        return min(("", ret_val or " "))

    def visit_create_column(self, create, first_pk: bool = ...) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_create_column(create, first_pk)
        return min(("", ret_val or " "))

    def visit_drop_table(self, drop) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_drop_table(drop)
        return min(("", ret_val or " "))

    def visit_ddl(self, ddl, **kwargs) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).visit_ddl(ddl, **kwargs)
        return min(("", ret_val or " "))

    def process(self, obj: Any, **kwargs: Any) -> str:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).process(obj, **kwargs)
        return min(("", ret_val or " "))

    def construct_params(self, params: Optional[Any] = ...) -> Any:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).construct_params(params)
        return ret_val

    def execute(self, *multiparams: Any, **params: Any) -> sa.engine.ResultProxy:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).execute(*multiparams, **params)
        return ret_val

    def scalar(self, *multiparams: Any, **params: Any) -> Any:
        """Insert DocString Here."""
        ret_val = super(DataflexDDLCompiler, self).scalar(*multiparams, **params)
        return ret_val


# noinspection SqlNoDataSourceInspection,PyMethodMayBeStatic, PyArgumentList,PyUnresolvedReferences
class DataflexCompiler(SQLCompiler):  # sa.sql.compiler.
    """Dataflex Compiler."""

    deferred: Set[str] = set()
    operators = sa.sql.compiler.OPERATORS.copy()
    operators[sa.sql.compiler.operators.collate] = ""
    operators[sa.sql.compiler.operators.concat_op] = " + "

    supported_functions = {
        # String Functions
        "ASCII",
        "CHAR",
        "CONCAT",
        "DIFFERENCE",
        "INSERT",
        "LCASE",
        "LEFT",
        "LENGTH",
        "LOCATE",
        "LTRIM",
        "REPEAT",
        "RIGHT",
        "RTRIM",
        "SOUNDEX",
        "SPACE",
        "SUBSTRING",
        "UCASE",
        # Numeric Functions
        "ABS",
        "ACOS",
        "ASIN",
        "ATAN",
        "ATAN2",
        "CEILING",
        "COS",
        "COT",
        "DEGREES",
        "EXP",
        "FLOOR",
        "LOG",
        "LOG10",
        "MOD",
        "PI",
        "POWER",
        "RADIANS",
        "RAND",
        "ROUND",
        "SIGN",
        "SIN",
        "SQRT",
        "TAN",
        "TRUNCATE",
        # Time and Date Functions
        "CURDATE",
        "CURTIME",
        "DAYNAME",
        "DAYOFMONTH",
        "DAYOFWEEK",
        "DAYOFYEAR",
        "HOUR",
        "MINUTE",
        "MONTH",
        "MONTHNAME",
        "NOW",
        "QUARTER",
        "SECOND",
        "TIMESTAMPADD",
        "TIMESTAMPDIFF",
        "WEEK",
        "YEAR",
        # System Functions
        "DATABASE",
        "IFNULL",
        "USER",
    }
    function_remaps = {
        # "COALESCE": "",
        "CURRENT_DATE": "CURDATE",
        "CURRENT_TIME": "CURTIME",
        "CURRENT_TIMESTAMP": "NOW",
        "CURRENT_USER": "USER",
        "LOCALTIME": "CURTIME",
        "LOCALTIMESTAMP": "NOW",
        "LOWER": "LCASE",
        "UPPER": "UCASE",
        "SYSDATE": "CURDATE",
        "SESSION_USER": "USER",
        # "ROLLUP": "",
        # "GROUPING SETS": "",
    }
    supported_convert_types = {
        "SQL_BIT",
        "SQL_CHAR",
        "SQL_VARCHAR",
        "SQL_DECIMAL",
        "SQL_NUMERIC",
        "SQL_REAL",
        "SQL_FLOAT",
        "SQL_DOUBLE",
        "SQL_BINARY",
        "SQL_VARBINARY",
        "SQL_SMALLINT",
        "SQL_INTEGER",
        "SQL_TINYINT",
        "SQL_TYPE_DATE",
        "SQL_TYPE_TIME",
        "SQL_TYPE_TIMESTAMP",
    }

    insert_single_values_expr: Any

    def _render_string_type(self, type_, name):
        assert self is not None  # Keep pylint happy
        text = name
        if type_.length:
            text += "(%d)" % type_.length
        if type_.collation:
            text += f' "{type_.collation}"'
        return text

    def _use_top(self, select):  # SQLA_1.4+
        assert self is not None  # Keep pylint happy
        simple_int_clause = getattr(select, "_simple_int_clause", lambda value: value is not None)

        return any(
            (
                all((select._offset_clause is None, getattr(select, "_limit_clause", None) is not None,)),
                all((simple_int_clause(getattr(select, "_limit_clause", None)), not select._offset)),
            )
        )

    def _get_limit_or_fetch(self, select):  # SQLA_1.4+
        if select._fetch_clause is None:
            return select._limit_clause
        else:
            return select._fetch_clause

    def _quote_tok(self, name: Any, tok: Any) -> str:
        """To quote or not to quote, that is the question."""
        if any((self.preparer._requires_quotes_illegal_chars(tok), isinstance(name, sa.sql.elements.quoted_name))):
            tok = self.preparer.quote(tok)
        return tok

    def get_select_precolumns(self, select, **kw):
        """FlexODBC uses TOP, similar to MS Access."""

        ret_val = super(DataflexCompiler, self).get_select_precolumns(select, **kw)

        if select._offset:
            raise NotImplementedError("FlexODBC does not support OFFSET")

        if getattr(select, "_limit_clause", None) is not None:
            ret_val = f"{ret_val}TOP {self.process(select._limit_clause, **kw)} "

        # (plagiarized from SQLAlchemy-Access)
        # try:
        #     if all(
        #         (
        #             hasattr(select, "_simple_int_limit"),
        #             getattr(select, "_simple_int_limit"),
        #             hasattr(select, "_limit"),
        #             getattr(select, "_limit"),
        #         )  # SQLA_1.3
        #     ):
        #         # ODBC drivers and possibly others
        #         # don't support bind params in the SELECT clause on SQL Server.
        #         # so have to use literal here.
        #         return f"{ret_val}TOP {int(select._limit)} "
        # except sa.exc.CompileError:
        #     pass
        #
        # if all((getattr(select, "_has_row_limiting_clause", False), self._use_top(select))):  # SQLA_1.4
        #     # ODBC drivers and possibly others
        #     # don't support bind params in the SELECT clause on SQL Server.
        #     # so have to use literal here.
        #     kw["literal_execute"] = True
        #     return f"{ret_val}TOP {self.process(self._get_limit_or_fetch(select), **kw)} "

        return ret_val

    def limit_clause(self, select, **kw):
        """FlexODBC doesn't support limit."""
        return ""

    def group_by_clause(self, select, **kw):
        """Emit properly formatted GROUP BY clauses."""

        kw.update({"literal_binds": True, "for_group_by": True})

        ret_val = ""
        asc_desc = ""
        compiled = select._group_by_clause._compiler_dispatch(self, **kw)

        if compiled:
            if any((compiled.casefold().endswith(" asc"), compiled.casefold().endswith(" desc"))):
                asc_desc = compiled[compiled.rfind(" ") :]

            parts = list(
                filter(
                    None,
                    map(
                        lambda item: "".join((x if x not in whitespace.replace(" ", "") else "" for x in item)),
                        map(
                            lambda part: part.split(asc_desc or chr(128293)),  # Unicode character 128293 is 🔥
                            chain.from_iterable(map(lambda entry: entry.split(" - "), compiled.split(" + "))),
                        ),
                    ),
                )
            )
            group_by = ", ".join(parts)
            group_by += f" {asc_desc}"
            ret_val = f" GROUP BY {group_by}"

        return ret_val

    def order_by_clause(self, select, **kw):
        """Emit properly formatted ORDER BY clauses."""

        kw.update({"literal_binds": True, "for_order_by": True})

        ret_val = ""
        asc_desc = ""
        compiled = select._order_by_clause._compiler_dispatch(self, **kw)

        if compiled:
            if any((compiled.casefold().endswith(" asc"), compiled.casefold().endswith(" desc"))):
                asc_desc = compiled[compiled.rfind(" ") :]

            parts = list(
                filter(
                    None,
                    map(
                        lambda item: "".join((x if x not in whitespace.replace(" ", "") else "" for x in item)),
                        map(
                            lambda part: part.split(asc_desc or chr(128293)),  # Unicode character 128293 is 🔥
                            chain.from_iterable(map(lambda entry: entry.split(" - "), compiled.split(" + "))),
                        ),
                    ),
                )
            )
            order_by = ", ".join(parts)

            if order_by.count(",") == 1:
                order_by = order_by.split(", ")[0]

            order_by += f" {asc_desc}"

            ret_val = f" ORDER BY {order_by}"

        return ret_val

    def visit_function(self, func, add_to_result_map=None, **kwargs) -> str:
        """Emit properly formatted function clauses.

        FlexODBC functions use {fn function_name(parameters)} as their syntax.

        Example:
            SELECT
              "customer"."name",
              {fn LEFT("customer"."name", 5)} AS "left_5"
            FROM
              "customer"
        """

        if add_to_result_map is not None:
            add_to_result_map(func.name, func.name, (), func.type)

        disp = getattr(self, f"visit_{func.name.lower()}_func", None)

        if disp:
            return disp(func, **kwargs)

        arg_spec = ""

        if func._has_args:
            arg_spec = list(iter(self.function_argspec(func, **kwargs)))

            if (arg_spec or [""])[0] == "(":
                (arg_spec or [""])[0] = ""

            if (arg_spec or [""])[-1] == ")":
                (arg_spec or [""])[-1] = ""

            arg_spec = "".join(arg_spec)

        if kwargs.get("for_group_by", kwargs.get("for_order_by", False)):
            return arg_spec

        func_name = self.function_remaps.get(func.name.upper(), func.name.upper()).upper()
        toks = [self._quote_tok(name=func_name, tok=tok) for tok in func.packagenames]

        if func_name in self.supported_functions:
            func_name = "".join(("{fn ", func_name, "(%(expr)s)}"))
        else:
            func_name += "(%(expr)s)"

        toks.append(func_name)

        ret_val = ".".join(toks)
        ret_val %= {"expr": arg_spec}

        return ret_val

    def visit_label(
        self,
        label,
        add_to_result_map=None,
        within_label_clause=False,
        within_columns_clause=False,
        render_label_as_label=None,
        **kw,
    ):
        """Emit properly formatted labels."""
        # only render labels within the columns clause
        # or ORDER BY clause of a select.  dialect-specific compilers
        # can modify this behavior.
        render_label_with_as = within_columns_clause and not within_label_clause
        render_label_only = render_label_as_label is label
        label_name = ""

        if render_label_only or render_label_with_as:
            if isinstance(label.name, sa.sql.elements._truncated_label):
                label_name = self._truncated_identifier("colident", label.name)
            else:
                label_name = label.name

        if render_label_with_as:
            if add_to_result_map is not None:
                add_to_result_map(
                    label_name, label.name, (label, label_name) + label._alt_names, label.type,
                )

            element = label.element._compiler_dispatch(
                self, within_columns_clause=True, within_label_clause=True, **kw
            )
            operator = self.operators[sa.sql.compiler.operators.as_]
            formatted_label = self.preparer.format_label(label, label_name)

            ret_val = "".join((element, operator, formatted_label))
        elif render_label_only:
            ret_val = self.preparer.format_label(label, label_name)
        else:
            ret_val = label.element._compiler_dispatch(self, within_columns_clause=False, **kw)

        return ret_val

    def visit_table(
        self, table, asfrom=False, iscrud=False, ashint=False, fromhints=None, use_schema=True, **kwargs
    ) -> str:
        """Emit properly formatted table names."""
        ret_val = super(DataflexCompiler, self).visit_table(
            table, asfrom, iscrud, ashint, fromhints, use_schema, **kwargs
        )
        return ret_val

    def visit_clauselist(self, clauselist, **kw):
        """Emit properly formatted clauses."""
        sep = clauselist.operator
        if sep is None:
            sep = " "
        else:
            sep = self.operators[clauselist.operator]

        text = sep.join(s for s in (c._compiler_dispatch(self, **kw) for c in clauselist.clauses) if s)
        if clauselist._tuple_values and self.dialect.tuple_in_values:
            text = "VALUES " + text
        return text

    def visit_unary(self, unary, **kw):
        """Emit properly formatted unary clauses."""
        if unary.operator:
            if unary.modifier:
                raise sa.exc.CompileError("Unary expression does not support operator and modifier simultaneously")
            disp = self._get_operator_dispatch(unary.operator, "unary", "operator")
            if disp:
                ret_val = disp(unary, unary.operator, **kw)
            else:
                ret_val = self._generate_generic_unary_operator(unary, self.operators[unary.operator], **kw)
        elif unary.modifier:
            disp = self._get_operator_dispatch(unary.modifier, "unary", "modifier")
            if disp:
                ret_val = disp(unary, unary.modifier, **kw)
            else:
                ret_val = self._generate_generic_unary_modifier(unary, self.operators[unary.modifier], **kw)
        else:
            raise sa.exc.CompileError("Unary expression has no operator or modifier")

        if ret_val.casefold().startswith("exists"):
            table_names = set(
                filter(None, (getattr(table, "name", None) for table in unary.element.locate_all_froms()))
            ) or {"FLEXERRS"}
            ret_val = f"\u0192{table_names.pop()}\u0192 {ret_val}"

        return ret_val

    def visit_binary(self, binary, override_operator=None, eager_grouping=False, **kw):
        """Emit properly formatted binary clauses."""
        # don't allow "? = ?" to render
        if (
            self.ansi_bind_rules
            and isinstance(binary.left, sa.sql.elements.BindParameter)
            and isinstance(binary.right, sa.sql.elements.BindParameter)
        ):
            kw["literal_binds"] = True

        operator_ = override_operator or binary.operator
        disp = self._get_operator_dispatch(operator_, "binary", None)
        if disp:
            return disp(binary, operator_, **kw)
        else:
            try:
                op_string = self.operators[operator_]
            except KeyError as err:
                sa.util.raise_(
                    sa.exc.UnsupportedCompilationError(self, operator_), replace_context=err,
                )
            else:
                return self._generate_generic_binary(binary, op_string, **kw)

    def visit_like_op_binary(self, binary, operator, **kw):
        """Emit properly formatted LIKE comparisons."""
        escape = binary.modifiers.get("escape", None)

        left = binary.left.compile(dialect=self.dialect, compile_kwargs={"literal_binds": True}).string
        right = binary.right.compile(dialect=self.dialect, compile_kwargs={"literal_binds": True}).string

        ret_val = f"{left} LIKE {right}".replace("' + '", "")

        # Base: SELECT "some_table"."id", "some_table"."data" FROM "some_table" WHERE
        # SQLA: ("some_table"."data" LIKE '%' || 'b/%cde' || '%' ESCAPE '/')
        # Emitted: ("some_table"."data" LIKE '%b\%cde%')
        # Desired: ("some_table"."data" LIKE '%b_cde%' AND {fn LOCATE('%', "some_table"."data")} > 0)

        # TODO: Document how the code below emulates escapement for `LIKE` clauses

        if escape:
            escape = self.render_literal_value(escape, sa.sql.sqltypes.STRINGTYPE)
            escape = escape.replace("'", "") if set(iter(escape)) != {"'"} else None
            if escape:
                escape_seqs = {f"{escape}%", f"{escape}_", f"{escape}{escape}"}
                escape_chars = {37, 39, 95, ord(escape)}
                ret_val = ret_val.replace("''", "\u0192")
                escape_positions = list(
                    map(
                        lambda pair: pair[0],
                        filter(
                            lambda entry: ret_val[entry[0] : entry[0] + int(2 * int(entry[0] + 1 <= len(ret_val)))]
                            in escape_seqs,
                            enumerate(ret_val),
                        ),
                    )
                )
                for pos in escape_positions:
                    back = "".join(takewhile(lambda entry: ord(entry) not in escape_chars, ret_val[pos - 1 :: -1]))[
                        ::-1
                    ]
                    forward = "".join(takewhile(lambda entry: ord(entry) not in escape_chars, ret_val[pos + 1 :]))
                    sequence = ret_val[pos + 1].join((back, forward)).replace("\u0192", "''")
                    ret_val = "".join(
                        (
                            "" if pair[0] == pos else "_" if pair[0] - 1 == pos and pair[1] == "%" else pair[1]
                            for pair in enumerate(ret_val)
                        )
                    )
                    ret_val += " AND {fn LOCATE('%(seq)s', %(left)s)} > 0" % {"seq": sequence, "left": left}
                ret_val = ret_val.replace("\u0192", "''")

        return ret_val

    def visit_collation(self, element, **kw: Any) -> str:
        """Emit properly formatted COLLATE clauses."""
        ret_val = super(DataflexCompiler, self).visit_collation(element, **kw)
        return min(("", ret_val or " "))

    def visit_cast(self, cast, **kwargs: Any) -> str:
        """Emit properly formatted CAST clauses."""

        kwargs.update({"literal_binds": True})

        clause = cast.clause._compiler_dispatch(self, **kwargs)
        type_clause = cast.typeclause._compiler_dispatch(self, **kwargs)
        sql_type = f"SQL_{type_clause.split('(')[0]}"
        ret_val = clause

        if sql_type in self.supported_convert_types:
            ret_val = "".join(("{fn CONVERT(", f"{clause}, {sql_type})", "}"))

        return ret_val

    def visit_case(self, clause, **kwargs: Any) -> str:
        """Emit properly formatted case clauses."""
        # Note: FlexODBC doesn't support CASE
        ret_val = super(DataflexCompiler, self).visit_case(clause, **kwargs)
        return min(("", ret_val))

    def visit_empty_set_expr(self, element_types):
        """Emit properly formatted empty-set expressions."""

        if element_types:
            del element_types  # Keep linters happy

        # This method *should* result in the compiler instance ultimately emitting a query that looks
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
        # the expected result. To work around these issues, the logic below tries to generate
        # sub-queries that amount to the same thing as though a true empty set was
        # supplied and correctly parsed. For example:
        #
        # SELECT "some_table"."id" FROM "some_table"
        # WHERE "some_table"."x" IN (SELECT MAX("some_table"."x") + 1 FROM "some_table")
        # ORDER BY "some_table"."id"
        #
        # Wherein this method is responsible for generating the `SELECT MAX("some_table"."x") + 1`
        # clause, thereby creating an unsatisfiable condition. This works for almost all of the testcases
        # for this method, with the exception of anything that relies on the empty set effectively
        # evaluating as NULL or FALSE (like using an empty set in a CASE statement). At the time of this
        # writing, (with the sole possible exception of the ANSI SQL function `IFNULL`, which FlexODBC
        # *does* support) no workable solution or workaround that could emulate the functionality of CASE
        # or IFF has presented itself. Until one is devised, this method should be considered incomplete
        # and unreliable.

        where = getattr(getattr(self, "statement", None), "_whereclause", None)
        lefts = (
            getattr(getattr(getattr(where, "left", None), "table", None), "name", None),
            getattr(getattr(where, "left", None), "key", None),
            getattr(getattr(getattr(where, "left", None), "type", None), "python_type", type(None)),
        )
        rights = (
            getattr(getattr(getattr(where, "right", None), "table", None), "name", None),
            getattr(getattr(where, "right", None), "key", None),
            getattr(getattr(getattr(where, "right", None), "type", None), "python_type", type(None)),
        )

        if all(lefts) and not all(rights):
            table_name, col_name, col_type = lefts
        elif not all(lefts) and all(rights):
            table_name, col_name, col_type = rights
        else:
            tables = set(
                filter(
                    None,
                    (
                        getattr(table, "name", None)
                        for table in getattr(getattr(self, "statement", None), "froms", list())
                    ),
                )
            )
            table_name = (tables or {"FLEXERRS"}).pop()
            col = next(iter(self.statement.c or list()), None)
            col_name = getattr(col, "name", None)
            col_type = getattr(getattr(col, "type", None), "python_type", type(None))

        if all(
            (
                isinstance(col_type(), Number),
                getattr(self.statement, "bind", None) is not None,
                hasattr(getattr(self.statement, "bind", None), "engine"),
            )
        ):
            query = f'''SELECT MAX("{col_name or "RECORD NUMBER"}") + 1 FROM "{table_name or "FLEXERRS"}"'''
            ret_val = str(next(iter(self.statement.bind.engine.execute(query).fetchone() or []), "NULL"))
        else:
            ret_val = f"'{uuid4()}'"

        return ret_val

    def visit_select(
        self,
        select,
        asfrom=False,
        parens=True,
        fromhints=None,
        compound_index=0,
        nested_join_translation=False,
        select_wraps_for=None,
        lateral=False,
        **kwargs,
    ) -> str:
        """Emit properly formatted select statements."""
        ret_val = super().visit_select(
            select,
            asfrom,
            parens,
            fromhints,
            compound_index,
            nested_join_translation,
            select_wraps_for,
            lateral,
            **kwargs,
        )
        table_names = list()

        if "\u0192" in ret_val:
            ret_val = ret_val.replace("\n", "")
            parts = ret_val.split(" ")
            table_names.extend(
                map(
                    lambda entry: entry.replace("\u0192", ""),
                    filter(lambda item: all((item.startswith("\u0192"), item.endswith("\u0192"))), parts),
                )
            )
            ret_val = " ".join(
                filter(lambda item: all((not item.startswith("\u0192"), not item.endswith("\u1092"))), parts)
            )

        if table_names and ret_val.casefold().count("where exists") == len(table_names):
            temp = ret_val.split(" WHERE EXISTS ")
            ret_val = ""
            base = list(iter(range(len(temp))))
            base_pairs = [iter(base)] * 2
            pos_pairs = list(zip_longest(*base_pairs, fillvalue=None))

            for pos, name in enumerate(table_names):
                start, end = pos_pairs[pos]
                ret_val += f"{temp[start]} FROM {name} WHERE EXISTS {temp[end]}"

        return ret_val

    def visit_insert(self, insert_stmt, asfrom=False, **kw) -> str:
        """Emit properly formatted INSERT statements."""

        toplevel = not self.stack

        self.stack.append({"correlate_froms": set(), "asfrom_froms": set(), "selectable": insert_stmt})

        crud_params = sa.sql.crud._setup_crud_params(self, insert_stmt, sa.sql.crud.ISINSERT, **kw)

        if not crud_params and not self.dialect.supports_default_values and not self.dialect.supports_empty_insert:
            raise sa.exc.CompileError(
                "The '%s' dialect with current database "
                "version settings does not support empty "
                "inserts." % self.dialect.name
            )

        if insert_stmt._has_multi_parameters:
            if not self.dialect.supports_multivalues_insert:
                raise sa.exc.CompileError(
                    "The '%s' dialect with current database "
                    "version settings does not support "
                    "in-place multirow inserts." % self.dialect.name
                )
            crud_params_single = crud_params[0]
        else:
            crud_params_single = crud_params

        preparer = self.preparer
        supports_default_values = self.dialect.supports_default_values

        text = "INSERT "

        if insert_stmt._prefixes:
            text += self._generate_prefixes(insert_stmt, insert_stmt._prefixes, **kw)

        text += "INTO "
        table_text = preparer.format_table(insert_stmt.table)

        if insert_stmt._hints:
            _, table_text = self._setup_crud_hints(insert_stmt, table_text)

        text += table_text

        if crud_params_single or not supports_default_values:
            text += " (%s)" % ", ".join([preparer.format_column(c[0]) for c in crud_params_single])

        if self.returning or insert_stmt._returning:
            returning_clause = self.returning_clause(insert_stmt, self.returning or insert_stmt._returning)

            if self.returning_precedes_values:
                text += " " + returning_clause
        else:
            returning_clause = None

        if insert_stmt.select is not None:
            query = insert_stmt.select.compile(dialect=self.dialect, compile_kwargs={"literal_binds": True}).string

            try:
                result = insert_stmt.select.bind.engine.execute(query).fetchall()
            except (AttributeError, pyodbc.Error):
                result = list()

            if not result:
                return f"{text} VALUES ({query})"

            params = dict(zip(insert_stmt.parameters.keys(), result[0]))
            text = (
                insert_stmt.table.insert()
                .values(**params)
                .compile(dialect=self.dialect, compile_kwargs={"literal_binds": True})
                .string
            )

            if len(result) > 1:
                for param_set in result[1:]:
                    deferred_params = dict(zip(insert_stmt.parameters.keys(), param_set))
                    deferred_insert = (
                        insert_stmt.table.insert()
                        .values(**deferred_params)
                        .compile(dialect=self.dialect, compile_kwargs={"literal_binds": True})
                        .string
                    )
                    self.deferred.add(deferred_insert)
            return text

        elif not crud_params and supports_default_values:
            text += " DEFAULT VALUES"
        elif insert_stmt._has_multi_parameters:
            text += " VALUES %s" % (
                ", ".join("(%s)" % (", ".join(c[1] for c in crud_param_set)) for crud_param_set in crud_params)
            )
        else:
            insert_single_values_expr = ", ".join([c[1] for c in crud_params])
            text += " VALUES (%s)" % insert_single_values_expr
            if toplevel:
                self.insert_single_values_expr = insert_single_values_expr

        if insert_stmt._post_values_clause is not None:
            post_values_clause = self.process(insert_stmt._post_values_clause, **kw)
            if post_values_clause:
                text += " " + post_values_clause

        if returning_clause and not self.returning_precedes_values:
            text += " " + returning_clause

        if self.ctes and toplevel and not self.dialect.cte_follows_insert:
            text = self._render_cte_clause() + text

        self.stack.pop(-1)

        if asfrom:
            return "(" + text + ")"

        return text

    def delete_table_clause(self, delete_stmt, from_table, extra_froms):
        """Delete table clause."""
        ret_val = from_table._compiler_dispatch(self, asfrom=True, iscrud=True).replace(" ", "_")
        return ret_val

    def process(self, obj: Any, **kwargs: Any) -> str:
        """Process the supplied object."""
        ret_val = obj._compiler_dispatch(self, **kwargs)
        return ret_val


# noinspection PyArgumentList,PyUnresolvedReferences
class DataflexExecutionContext(DefaultExecutionContext):
    """Dataflex Execution Context."""

    def get_lastrowid(self):
        """Get the id of the last inserted row."""
        super(DataflexExecutionContext, self).get_lastrowid()


# noinspection PyArgumentList,PyUnresolvedReferences
class DataflexDialect(DefaultDialect):
    """Dataflex Dialect."""

    name = "dataflex"
    dbapi = pyodbc

    # Supported parameter styles: ["qmark", "numeric", "named", "format", "pyformat"]
    default_paramstyle = "qmark"
    dbapi.paramstyle = default_paramstyle

    poolclass = sa.pool.SingletonThreadPool
    statement_compiler = DataflexCompiler
    ddl_compiler = DataflexDDLCompiler
    type_compiler = DataflexTypeCompiler
    preparer = DataflexIdentifierPreparer
    execution_ctx_cls = DataflexExecutionContext

    max_identifier_length = 20
    max_index_name_length = 20

    postfetch_lastrowid = False
    implicit_returning = False
    inline_comments = False

    supports_alter = False
    supports_views = False
    supports_comments = False
    supports_sequences = False
    supports_native_enum = False
    supports_empty_insert = False
    supports_native_boolean = False
    supports_sane_rowcount = False
    supports_for_update_of = False
    supports_default_values = False
    supports_native_decimal = False
    supports_is_distinct_from = False
    supports_right_nested_joins = False
    supports_multivalues_insert = False
    supports_sane_multi_rowcount = False
    supports_server_side_cursors = False
    supports_simple_order_by_label = False

    sequences_optional = False
    preexecute_autoincrement_sequences = False
    returns_unicode_strings = True
    supports_unicode_binds = True
    supports_unicode_statements = True
    requires_name_normalize = False

    odbc_type_map = {
        # SQL Type Code -> Mapped Type  # actual odbc type
        -10: LongVarChar,  # long unicode varchar
        -9: LongVarChar,  # unicode varchar
        -8: LongVarChar,  # unicode char
        -7: Logical,  # bit
        -6: Integer,  # tinyint
        -5: BigInt,  # bigint
        -4: LongVarChar,  # long varbinary
        -3: LongVarChar,  # varbinary (bit-varying datatype)
        -2: LongVarChar,  # binary (bit datatype)
        -1: LongVarChar,  # long varchar
        1: Char,  # char
        2: DoublePrecision,  # numeric
        3: Decimal,  # decimal
        4: Integer,  # integer
        5: Integer,  # smallint
        6: DoublePrecision,  # float
        7: DoublePrecision,  # real
        8: DoublePrecision,  # double precision
        9: Date,  # date
        10: Time,  # time
        11: Timestamp,  # timestamp
        12: VarChar,  # varchar
    }

    @staticmethod
    def _check_unicode_returns(*args: Any, **kwargs: Any):
        """Check if the local system supplies unicode returns."""

        if args and kwargs:
            del args, kwargs  # Keep linters happy

        # The driver should pretty much always be running on a modern
        # Windows system, so it's more or less safe to assume we'll
        # always get a unicode string back for string values
        return True

    @sa.engine.reflection.cache
    def has_table(self, connection, table_name, schema=None, **kw):
        """Check the existence of a particular table in the database.

        Given a :class:`_engine.Connection` object and a string
        `table_name`, return True if the given table (possibly within
        the specified `schema`) exists in the database, False
        otherwise.
        """
        table_name = table_name.replace(" ", "_")
        try:
            return connection.engine.raw_connection().cursor().tables(table_name).fetchone() is not None
        except pyodbc.Error:
            return False

    @sa.engine.reflection.cache
    def get_table_names(self, connection, schema=None, **kw):
        """Get the names of all the local tables."""

        # pyodbc table objects have the following properties:
        #
        # table_cat: The catalog name.
        # table_schem: The schema name.
        # table_name: The table name.
        # table_type: One of TABLE, VIEW, SYSTEM TABLE, GLOBAL TEMPORARY, LOCAL TEMPORARY, ALIAS, SYNONYM,
        #             or a data source-specific type name.
        # remarks: A description of the table.

        cursor = connection.engine.raw_connection().cursor()

        try:
            table_names = {table.table_name for table in cursor.tables(tableType="TABLE").fetchall()}
            if table_names:
                return list(table_names)
        except pyodbc.Error:
            pass

        return []

    @sa.engine.reflection.cache
    def get_view_names(self, connection, schema=None, **kw):
        """Get the names of all local views."""
        cursor = connection.engine.raw_connection().cursor()

        try:
            view_names = {view.table_name for view in cursor.tables(tableType="VIEW").fetchall()}
            if view_names:
                return list(view_names)
        except pyodbc.Error:
            pass

        return []

    def get_columns(self, connection, table_name, schema=None, **kw):
        """Get the column names and data-types for a given table."""

        table_name = table_name.replace(" ", "_")
        table_data = dict()
        conn = connection.engine.raw_connection()
        cursor = conn.cursor()
        result = []

        try:
            for column in cursor.columns(table=table_name).fetchall():
                table_data[column.column_name] = {
                    "catalog": getattr(column, "table_cat", None),
                    "table_name": getattr(column, "table_name", None),
                    "column_name": getattr(column, "column_name", None),
                    "data_type": getattr(column, "data_type", None),
                    "type_name": getattr(column, "type_name", None),
                    "column_size": getattr(column, "column_size", None),
                    "buffer_length": getattr(column, "buffer_length", None),
                    "decimal_digits": getattr(column, "decimal_digits", None),
                    "num_prec_radix": getattr(column, "num_prec_radix", None),
                    "nullable": getattr(column, "nullable", None),
                    "remarks": getattr(column, "remarks", None),
                    "default_value": getattr(column, "column_def", None),
                    "sql_data_type": getattr(column, "sql_data_type", None),
                    "ordinal_position": getattr(column, "ordinal_position", None),
                    "is_nullable": getattr(column, "is_nullable", None),
                }
                del column

            for column, data in table_data.items():
                column_class = self.odbc_type_map.get(cg(data, "sql_data_type", -1))
                column_type = column_class()
                if issubclass(column_class, (sa.types.String, sa.types.Text)):
                    column_type.length = data.get("column_size")
                elif issubclass(column_class, (sa.types.DECIMAL, sa.types.Float)):
                    column_type.precision = data.get("column_size")
                    column_type.scale = data.get("decimal_digits")
                result.append(
                    {
                        "name": column,
                        "type": column_type,
                        "nullable": bool(all((strtobool(data.get("nullable")), strtobool(data.get("is_nullable"))))),
                        "default": data.get("column_def"),
                        "autoincrement": column.casefold() in ("record_number", "id"),
                    }
                )
                del column, data
        except pyodbc.Error:
            pass

        return result

    def get_pk_constraint(self, connection, table_name, schema=None, *args: Any, **kwargs: Any):
        """ Return information about the primary key constraint on `table_name`.

            Given a :class:`_engine.Connection`, a string
            `table_name`, and an optional string `schema`, return primary
            key information as a dictionary with these keys:

            constrained_columns
              a list of column names that make up the primary key

            name
              optional name of the primary key constraint.
        """
        table_name = table_name.replace(" ", "_")
        conn = connection.engine.raw_connection()
        cursor = conn.cursor()
        pks = list()

        try:
            pks.extend(cursor.primaryKeys(table_name).fetchall())
        except pyodbc.Error:
            pass

        if not pks:
            return {
                "name": None,
                "constrained_columns": [],
            }

        pk_name = max(set((row[5] for row in pks)) or {"PRIMARY"})

        return {
            "name": pk_name,
            "constrained_columns": [row[3] for row in pks],
        }

    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        """Get the list of foreign keys from a given table."""

        # TheFlexODBC driver's support for foreign keys is non-existent
        # so it's *extremely* likely that this list will always be empty

        table_name = table_name.replace(" ", "_")
        conn = connection.engine.raw_connection()
        cursor = conn.cursor()
        fks = list()

        try:
            fks.extend([key[3] for key in cursor.foreignKeys(table_name)])
        except pyodbc.Error:
            pass

        return fks

    @sa.engine.reflection.cache
    def get_indexes(self, connection, table_name, *args, **kwargs):
        """Return information about indexes in `table_name`.

        Given a :class:`_engine.Connection`, a string
        `table_name` and an optional string `schema`, return index
        information as a list of dictionaries with these keys:

        name
          the index's name

        column_names
          list of column names in order

        unique
          boolean
        """

        catalog = kwargs.get("catalog", None)
        schema = kwargs.get("schema", None)
        unique = kwargs.get("unique", False)
        quick = kwargs.get("catalog", False)
        cursor = connection.engine.raw_connection().cursor()

        stat_cols = {
            "table_cat": 0,
            "table_schem": 1,
            "table_name": 2,
            "non_unique": 3,
            "index_qualifier": 4,
            "index_name": 5,
            "type": 6,
            "ordinal_position": 7,
            "column_name": 8,
            "asc_or_desc": 9,
            "cardinality": 10,
            "pages": 11,
            "filter_condition": 12,
        }

        index_data = dict()

        for pos, row in enumerate(
            cursor.statistics(table_name, catalog=catalog, schema=schema, unique=unique, quick=quick)
        ):
            index_data[f"index_{pos}"] = {key: row[value] for key, value in stat_cols.items()}

        index_names = {index["index_name"] for index in index_data.values()}

        ret_val = list()

        for index in index_names:
            unique = all(
                (
                    idx["non_unique"] == 0
                    for idx in filter(lambda entry: entry["index_name"] == index, index_data.values())
                )
            )
            column_names = [
                idx["column_name"] for idx in filter(lambda entry: entry["index_name"] == index, index_data.values())
            ]
            ret_val.append({"name": index, "unique": unique, "column_names": column_names})

        return sorted(
            filter(lambda entry: entry["name"] and entry["column_names"], ret_val),
            key=lambda entry: entry.get("name"),
        )

    @sa.engine.reflection.cache
    def get_temp_table_names(self, connection, schema=None, **kw):
        """Get the names of any extant temporary tables."""
        return []

    @staticmethod
    def get_temp_view_names(*args: Any, **kwargs: Any):
        """Get the names of any temporary views."""
        # DataFlex doesn't supply View functionality so this will always be empty
        return []

    @staticmethod
    def get_view_definition(*args, **kwargs):
        """Get the definition of a specific local view."""
        # DataFlex doesn't supply View functionality
        return {}

    @sa.engine.reflection.cache
    def get_unique_constraints(self, *args, **kwargs):
        r"""Return information about unique constraints in `table_name`.

        Given a string `table_name` and an optional string `schema`, return
        unique constraint information as a list of dicts with these keys:

        name
          the unique constraint's name

        column_names
          list of column names in order

        \**kw
          other options passed to the dialect's get_unique_constraints()
          method
        """
        indexes = self.get_indexes(*args, **kwargs)

        return list(filter(lambda index: index.get("unique", False) is True, indexes))

    @staticmethod
    def get_check_constraints(*args: Any, **kwargs: Any):
        """The FlexODBC driver doesn't really support constraints."""
        return []

    @sa.engine.reflection.cache
    def get_table_comment(self, connection, table_name, *args, **kwargs):
        r"""Return the "comment" for the table identified by `table_name`.

        Given a string `table_name` and an optional string `schema`, return
        table comment information as a dictionary with this key:

        text
           text of the comment

        Raises ``NotImplementedError`` for dialects that don't support
        comments.
        """

        if args:
            del args

        table_name = table_name.replace(" ", "_")
        catalog = kwargs.get("catalog", None)
        schema = kwargs.get("schema", None)
        table_type = kwargs.get("tableType", None)
        pyodbc_cursor = connection.engine.raw_connection().cursor()
        table_data = pyodbc_cursor.tables(
            table=table_name, catalog=catalog, schema=schema, tableType=table_type
        ).fetchone()
        comments = getattr(table_data, "remarks", "")

        return {"text": comments}

    @staticmethod
    def has_sequence(*args, **kwargs):
        """DataFlex doesn't support sequences, so it will never have a queried sequence."""
        return False

    def set_isolation_level(self, dbapi_conn, level):
        """Set isolation level."""
        super().set_isolation_level(dbapi_conn, level)

    def get_isolation_level(self, dbapi_conn):
        """Get isolation level."""
        super().get_isolation_level(dbapi_conn)

    def do_executemany(self, cursor, statement, parameters, context=None):
        """Insert DocString Here."""
        for num, param_set in enumerate(parameters):
            self.do_execute(cursor, statement, param_set, context)

    def do_execute(self, cursor, statement, parameters, context=None):
        """Insert DocString Here."""

        def log_statement(st, prms=tuple()):
            """Log the supplied about-to-be-executed statement."""
            from pathlib import Path

            statement_log = (Path().home() / "Downloads" / "statements.txt").resolve()
            if not isinstance(st, str):
                st = str(st)

            if statement_log.exists():
                try:
                    with statement_log.open("ab+") as writer:
                        st_ = st.replace("\n", " ").replace("\t", " ").replace("\r", " ").split(" ")
                        st_ = " ".join(filter(None, st_)).encode("utf8")
                        st_ = f"\nStatement: ".encode("utf8") + st_ + "\n".encode("utf8")
                        writer.write(st_)
                        if prms:
                            writer.write(f"Params: {prms}\n".encode("utf8"))
                except Exception as err:
                    print(f"{type(err)} -> {err}")

        # TODO: Workout how to add an arbitrary table to FROM-less SELECT queries
        if all(("select " in statement.casefold(), " from " not in statement.replace("\n", "").casefold())):
            statement = ""

        q_count = statement.count("?")

        if q_count == 0:
            parameters = ()
        elif q_count != len(parameters):
            parameters = parameters[:q_count]

        if "top ?" in statement.casefold() and len(parameters) >= 1:
            replacement = f"TOP {parameters[0]}"
            statement = statement.replace("top ?", replacement).replace("TOP ?", replacement)
            parameters = parameters[1:]

        if statement:
            log_statement(statement, parameters)
            cursor.execute(statement, parameters)

        while self.statement_compiler.deferred:
            statement = self.statement_compiler.deferred.pop()
            log_statement(statement)
            cursor.execute(statement)
