# This module is based on the SqlAlchemy-Access, which can be found here:
# https://github.com/sqlalchemy/sqlalchemy-access.git

"""
Support for DataFlex using the FlexODBC Driver from FlexData
"""


from sqlalchemy import sql, schema, types, exc, pool
from sqlalchemy.sql import compiler, expression
from sqlalchemy.engine import default, base, reflection
from sqlalchemy import processors
from sqlalchemy.sql.compiler import OPERATORS, FUNCTIONS, elements

from re import sub

from warnings import warn

from datetime import date, time, datetime
from decimal import Decimal

import pyodbc


# These are relatively simple, as the underlying DataFlex driver only *really* supports
# 'DECIMAL', 'CHAR', 'DATE', and 'INTEGER' datatypes using the v3 driver or
# 'DECIMAL', 'VARCHAR', 'DATE', and 'INTEGER' using the v4 driver


class dfDECIMAL(types.DECIMAL):
    __visit_name__ = "DECIMAL"


class dfCHAR(types.VARCHAR):
    __visit_name__ = "CHAR"


class dfDATE(types.DATE):
    __visit_name__ = "DATE"


class dfINTEGER(types.INTEGER):
    __visit_name__ = "INTEGER"


dfDecimal = dfDECIMAL
dfChar = dfCHAR
dfDate = dfDATE
dfInt = dfINTEGER

# These are the "release notes" detailing what the driver does and does not support
#
# Only ODBC Core Level SQL is supported.
# Only ODBC Level 2 API calls are supported.
# ODBC Version 3.0 compliant.
# Column and table names are not case-sensitive, string data comparisons are case sensitive.
# Character values supplied for parameterized queries (SELECT * FROM EMP WHERE NAME = ?) are limited to 255 characters.
# Transactions are not supported.
# Qualifiers and owners are not allowed on databases, tables, etc.
# Table creation and deletion are not supported currently.
# DataFlex Indexes are used when the ORDER BY columns matches an existing available index
# If a matching index is not found, the result will be sorted after it has been acquired from the file.


"""
Map names returned by the "type_name" column of pyodbc's Cursor.columns method to our dialect types.

These names are what you would retrieve from INFORMATION_SCHEMA.COLUMNS.DATA_TYPE if DataFlex
supported those types of system views.
"""
ischema_names = {
    "DECIMAL": dfDecimal,
    "VARCHAR": dfChar,
    "CHAR": dfChar,
    "DATE": dfDate,
    "INTEGER": dfInt,
}


class DataFlexExecutionContext(default.DefaultExecutionContext):
    # TODO: This probably needs to be changed to whatever would actually work for DF

    def get_lastrowid(self):
        self.cursor.execute("SELECT @@identity AS lastrowid")
        return self.cursor.fetchone()[0]


class DataFlexCompiler(compiler.SQLCompiler):

    # TODO: A lot of this class is relatively unchanged from SQLA-Access, but probably should be

    extract_map = compiler.SQLCompiler.extract_map.copy()
    extract_map.update(
        {
            "month": "m",
            "day": "d",
            "year": "yyyy",
            "second": "s",
            "hour": "h",
            "doy": "y",
            "minute": "n",
            "quarter": "q",
            "dow": "w",
            "week": "ww",
        }
    )

    def visit_cast(self, cast, **kw):
        return cast.clause._compiler_dispatch(self, **kw)

    def get_select_precolumns(self, select, **kw):
        # (plagiarized from mssql/base.py)
        """ Access puts TOP, it's version of LIMIT here """

        s = ""
        if select._distinct:
            s += "DISTINCT "

        if select._simple_int_limit and not select._offset:
            # ODBC drivers and possibly others
            # don't support bind params in the SELECT clause on SQL Server.
            # so have to use literal here.
            s += "TOP %d " % select._limit

        if s:
            return s
        else:
            return compiler.SQLCompiler.get_select_precolumns(self, select, **kw)

    def limit_clause(self, select, **kw):
        """Limit in access is after the select keyword"""
        return ""

    def binary_operator_string(self, binary):
        """Access uses "mod" instead of "%" """
        return binary.operator == "%" and "mod" or binary.operator

    def visit_concat_op_binary(self, binary, operator, **kw):
        return "%s & %s" % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    function_rewrites = {
        "current_date": "now",
        "current_timestamp": "now",
        "length": "len",
    }

    # The driver only supports these function calls

    supported_functions = [
        # String functions
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
        # Date & Time functions
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
        # System functions
        "DATABASE",
        "IFNULL",
        "USER",
    ]

    def visit_function(self, func, **kwargs):
        """DataFlex functions are barely supported and are invoked
        differently from most other SQL dialects. Rewrite the ones that are supported
        """

        # The FlexODBC driver requires function calls to be wrapped thusly:
        # {fn FUNCTION(parameter)}

        if func.name.upper() in self.supported_functions:
            disp = getattr(self, "visit_%s_func" % func.name.lower(), None)
            if disp:
                return "{fn " + str(disp(func, **kwargs)) + "}"
            else:
                name = next(
                    filter(
                        lambda x: x.upper() == func.name.upper(),
                        self.supported_functions,
                    ),
                    None,
                )
                if name:
                    if func._has_args:
                        name += "%(expr)s"
                else:
                    warn("Function not in list of supported functions!", RuntimeWarning)
                    name = func.name
                    name = (
                        self.preparer.quote(name)
                        if self.preparer._requires_quotes_illegal_chars(name)
                        or isinstance(name, elements.quoted_name)
                        else name
                    )
                    name += "%(expr)s"
                return (
                    "{fn "
                    + str(
                        ".".join(
                            [
                                (
                                    self.preparer.quote(tok)
                                    if self.preparer._requires_quotes_illegal_chars(tok)
                                    or isinstance(name, elements.quoted_name)
                                    else tok
                                )
                                for tok in func.packagenames
                            ]
                            + [name]
                        )
                        % {"expr": self.function_argspec(func, **kwargs)}
                    )
                    + "}"
                )
        else:
            return ""

    def for_update_clause(self, select, **kw):
        """FOR UPDATE is not supported by Access; silently ignore"""
        # This is unchanged from SQLA-Access, but I don't think DF actually supports
        # FOR UPDATE anyway, so there's no point in messing with it
        return ""

    # Strip schema
    def visit_table(self, table, asfrom=False, **kw):
        if asfrom:
            return self.preparer.quote(table.name)
        else:
            return ""

    def visit_join(self, join, asfrom=False, **kw):
        """Silently ignore JOINs
        """
        # TODO: DataFlex actually does support joins, but the syntax is finicky,
        #       so this will need to be revisited
        return ""

    def visit_extract(self, extract, **kw):
        field = self.extract_map.get(extract.field, extract.field)
        return 'DATEPART("%s", %s)' % (field, self.process(extract.expr, **kw))

    def visit_empty_set_expr(self, type_):
        literal = None
        repr_ = repr(type_[0])
        if repr_.startswith("Integer("):
            literal = "1"
        elif repr_.startswith("String("):
            literal = "'x'"
        elif repr_.startswith("NullType("):
            literal = "NULL"
        else:
            raise ValueError("Unknown type_: %s" % type(type_[0]))
        stmt = "SELECT %s FROM USysSQLAlchemyDUAL WHERE 1=0" % literal
        return stmt


class DataFlexTypeCompiler(compiler.GenericTypeCompiler):
    # The underlying DataFlex driver *really* doesn't support much
    # in the way of datatypes, so this may be entirely perfunctory
    #
    # It's being done anyway to keep this library as in-line with
    # sqlalchemy-access as possible, as this library is a shameless
    # ripoff of sqlalchemy-access

    def visit_DECIMAL(self, type_, **kw):
        return dfDecimal.__visit_name__

    def visit_CHAR(self, type_, **kw):
        return dfCHAR.__visit_name__

    def visit_DATE(self, type_, **kw):
        return dfDATE.__visit_name__

    def visit_INTEGER(self, type_, **kw):
        return dfINTEGER.__visit_name__


class DataFlexDDLCompiler(compiler.DDLCompiler):
    def get_column_specification(self, column, **kw):
        if column.table is None:
            raise exc.CompileError(
                # This message has only been update to say DataFlex instead of Access
                # I'm not sure that it's actually accurate
                "DataFlex requires Table-bound columns in order to generate DDL"
            )

        colspec = self.preparer.format_column(column)
        seq_col = column.table._autoincrement_column
        if seq_col is column:
            colspec += " COUNTER"
        else:
            colspec += " " + self.dialect.type_compiler.process(column.type)

            if column.nullable is not None and not column.primary_key:
                if not column.nullable or column.primary_key:
                    colspec += " NOT NULL"
                else:
                    colspec += " NULL"

            default = self.get_column_default_string(column)
            if default is not None:
                colspec += " DEFAULT " + default

        return colspec

    def visit_drop_index(self, drop):
        index = drop.element
        self.append(
            "\nDROP INDEX [%s].[%s]"
            % (index.table.name, self._index_identifier(index.name))
        )


class DataFlexIdentifierPreparer(compiler.IdentifierPreparer):
    reserved_words = compiler.RESERVED_WORDS.copy()

    # This list is taken directly from the FlexODBC v4 Help File
    reserved_words.update(
        [
            "ABSOLUTE",
            "ADA",
            "ADD",
            "ALL",
            "ABSOLUTE",
            "ADA",
            "ADD",
            "ALL",
            "ALLO",
            "ALT",
            "AND",
            "ANY",
            "AR",
            "AS",
            "ASC",
            "ASSER",
            "AT",
            "AUT",
            "AVG",
            "BE",
            "BETWEEN",
            "BIT",
            "BIT_L",
            "BY",
            "CASCADE",
            "CASCA",
            "CASE",
            "CAST",
            "CATALOG",
            "CHAR",
            "CHAR_",
            "CHARACTE",
            "CHARA",
            "CHECK",
            "CLOSE",
            "COALESC",
            "COBOL",
            "COLLATE",
            "COLLATION",
            "COLUMN",
            "COMMIT",
            "CONNECT",
            "CONNECT",
            "CONSTRAINT",
            "CONST",
            "CONTINU",
            "CONVERT",
            "CORRE",
            "COUNT",
            "CREATE",
            "CURRENT",
            "CURR",
            "CUR",
            "CURRENT_TIM",
            "CUR",
            "DATE",
            "DAY",
            "DEALLOCATE",
            "DEC",
            "DECIMA",
            "DECLA",
            "DEFERRAB",
            "DEFERRED",
            "DELETE",
            "DESC",
            "DESCRIBE",
            "DESCRIPTOR",
            "DIAGNOS",
            "DICTION",
            "DISCONN",
            "DISPLA",
            "DISTI",
            "DOM",
            "DOUBLE",
            "DROP",
            "ELSE",
            "END",
            "ENDE",
            "ESCAPE",
            "EXCEPT",
            "EXCEPTION",
            "EXEC",
            "EXECU",
            "EXISTS",
            "EXTER",
            "EXTRA",
            "FALS",
            "FETCH",
            "FIRST",
            "FLOAT",
            "FOR",
            "FORE",
            "FOR",
            "FOUND",
            "FRO",
            "FULL",
            "GET",
            "GLOBAL",
            "GO",
            "GOTO",
            "GRANT",
            "GROUP",
            "HAVING",
            "HO",
            "IDENTITY",
            "IGNOR",
            "IMMEDIATE",
            "IN",
            "INCLUD",
            "INDEX",
            "INDICATOR",
            "INITIALL",
            "INNER",
            "INPUT",
            "INSENSITIV",
            "INSE",
            "IN",
            "INTERSECT",
            "INTE",
            "INTO",
            "IS",
            "ISOLA",
            "JOIN",
            "KEY",
            "LANG",
            "LAST",
            "LEFT",
            "LEVEL",
            "LIKE",
            "LOC",
            "LOWER",
            "MATCH",
            "MAX",
            "MIN",
            "MINUTE",
            "MODULE",
            "MONTH",
            "MUMP",
            "NAME",
            "NAT",
            "NCHA",
            "NEXT",
            "NONE",
            "NOT",
            "NU",
            "NUL",
            "NU",
            "OCTE",
            "OF",
            "OFF",
            "ON",
            "ONLY",
            "OPEN",
            "OPTION",
            "OR",
            "ORDER",
            "OUTER",
            "OU",
            "OVERLAP",
            "PARTIAL",
            "PASCAL",
            "PLI",
            "POSITION",
            "PRECIS",
            "PREPARE",
            "PRESERVE",
            "PRIMAR",
            "PRIOR",
            "PRIVILEG",
            "PROCE",
            "PUBLIC",
            "RESTR",
            "REVOKE",
            "RIGHT",
            "ROLLBAC",
            "ROWS",
            "SELECT",
            "SEQUENCE",
            "SET",
            "SIZE",
            "SELECT",
            "SEQU",
            "SET",
            "SIZE",
            "SMALLINT",
            "SOME",
            "SQL",
            "SQLCA",
            "SQLCODE",
            "SQLE",
            "SQLSTAT",
            "SQLWA",
            "SUBSTRING",
            "SUM",
            "SYST",
            "TABLE",
            "TEMPORARY",
            "THEN",
            "TIM",
            "TIMESTAMP",
            "TIMEZONE_H",
            "TIMEZONE_MIN",
            "TO",
            "TRANS",
            "TRANSL",
            "TRANSLAT",
            "TRUE",
            "UNION",
            "UNIQUE",
            "UNKN",
            "UPDAT",
            "UPPER",
            "USAGE",
            "USER",
            "USING",
            "VALU",
            "VALUE",
            "VARCHAR",
            "VARYING",
            "VIEW",
            "WHEN",
            "WHEN",
        ]
    )

    def __init__(self, dialect):
        super(DataFlexIdentifierPreparer, self).__init__(
            dialect, initial_quote="", final_quote=""
        )


class DataFlexDialect(default.DefaultDialect):
    colspecs = {}
    name = "dataflex"

    supports_views = False
    supports_sequences = False
    supports_alter = False
    supports_comments = False
    inline_comments = False
    supports_right_nested_joins = False

    postfetch_lastrowid = False

    supports_sane_rowcount = False
    supports_sane_multi_rowcount = False
    supports_default_values = False
    supports_empty_insert = False
    supports_multivalues_insert = False
    supports_server_side_cursors = False

    reflection_options = ()

    supports_native_boolean = (
        True  # suppress CHECK constraint on boolean columns, which don't exist anyway
    )
    supports_simple_order_by_label = False
    _need_decimal_fix = False

    supports_is_distinct_from = False

    poolclass = pool.SingletonThreadPool
    statement_compiler = DataFlexCompiler
    ddl_compiler = DataFlexDDLCompiler
    type_compiler = DataFlexTypeCompiler
    preparer = DataFlexIdentifierPreparer
    execution_ctx_cls = DataFlexExecutionContext

    @classmethod
    def dbapi(cls):
        import pyodbc as module

        module.pooling = (
            False  # required for DataFlex databases with ODBC linked tables
        )
        return module

    def connect(self, *cargs, **cparams):
        # The DataFlex driver is apparently finicky about the
        # formatting if its connection string, so we'll have to
        # do some fiddling to the url that gets passed in
        fixed_url = list()
        for element in cargs[0].split(";"):
            if element[:7].upper() not in ["SERVER=", "TRUSTED"]:
                if "database=" in element.lower():
                    fixed_url.append(element.replace(element[:9], "DBQ="))
                else:
                    fixed_url.append(element)
        fixed_url = ";".join(fixed_url)

        # The DataFlex driver doesn't support being interrogated about
        # its autocommit capabilities, so it has to be explicitly set
        return self.dbapi.connect(fixed_url, autocommit=True)

    def create_connect_args(self, url, **kwargs):
        # Supported query arguments:
        # v3 - DBQ, DecimalOption, Driver, PWD, UID, YearDigits, DisplayRecNum
        # v4 - DBQ, DecimalOption, Driver, YearDigits, DisplayRecNum, UseSimulatedTransactions, ConvertToLongVARCHAR

        connectors = ["Driver={FlexODBCv3}"]

        if url.username is not None:
            connectors.append(f"UID={url.username}")
        if url.password is not None:
            connectors.append(f"UID={url.password}")
        if url.host is None and url.database is not None:
            connectors.append(f"DBQ={url.database}")
        elif url.host is not None and url.database is None:
            connectors.append(f"DBQ={url.host}")
        elif url.host is None and url.database is None:
            raise FileNotFoundError(
                "An absolute path to a DataFlex folder is required!"
            )
        else:
            connectors.append(f"DBQ={url.database}")
        if url.query.get("DecimalOption", None) is not None:
            connectors.append(f"DecimalOption={url.query.get('DecimalOption')}")
        if url.query.get("YearDigits", None) is not None:
            connectors.append(f"YearDigits={url.query.get('YearDigits')}")
        if url.query.get("DisplayRecNum", None) is not None:
            connectors.append(f"DisplayRecNum={url.query.get('DisplayRecNum')}")

        return [[";".join(connectors)], {"autocommit": True}]

    def _check_unicode_returns(self, connection, additional_tests=None):
        return True

    def has_table(self, connection, tablename, schema=None):
        pyodbc_crsr = connection.engine.raw_connection().cursor()
        try:
            result = pyodbc_crsr.tables(table=tablename).fetchone()
            return bool(result)
        except Exception:
            return False

    @reflection.cache
    def get_table_names(self, connection, schema=None, **kw):
        pyodbc_crsr = connection.engine.raw_connection().cursor()
        table_names = list(
            set([x.tablename for x in pyodbc_crsr.tables(tableType="TABLE").fetchall()])
        )
        vetted_table_names = list()
        for table in table_names:
            try:
                pyodbc_crsr.execute(f"SELECT * FROM {table}").fetchone()
                vetted_table_names.append(table)
            except Exception:
                pass
        return vetted_table_names

    @reflection.cache
    def get_view_names(self, connection, schema=None, **kw):
        return []  # DataFlex doesn't support views

    def get_columns(self, connection, table_name, schema=None, **kw):
        pyodbc_cnxn = connection.engine.raw_connection()
        pyodbc_crsr = pyodbc_cnxn.cursor()
        result = list()
        for row in pyodbc_crsr.columns(table=table_name):
            class_ = ischema_names[row.typename]
            type_ = class_()
            if class_ is types.String:
                type_.length = row.length
            elif class_ in [types.DECIMAL, types.Numeric]:
                type_.precision = row.precision
                type_.scale = row.scale
            result.append(
                {
                    "name": row.columnname,
                    "type": type_,
                    "nullable": bool(row.nullable),
                    "default": None,  # DataFlex doesn't really provide a "default"
                    "autoincrement": (row.typename == "COUNTER"),
                }
            )
        return result

    def get_primary_keys(self, connection, table_name, schema=None, **kw):
        return []  # DataFlex doesn't *really* support primary keys

    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        return []  # DataFlex doesn't *really* support primary keys

    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        return (
            []
        )  # Ahahahaha. Fun fact, DataFlex doesn't really support foreign keys either

    def get_indexes(self, connection, table_name, schema=None, **kw):
        pyodbc_crsr = connection.engine.raw_connection().cursor()
        indexes = {}
        for row in pyodbc_crsr.statistics(table_name).fetchall():
            if row.indexname is not None:
                if row.indexname in indexes:
                    indexes[row.indexname]["column_names"].append(row.columnname)
                else:
                    indexes[row.indexname] = {
                        "name": row.indexname,
                        "unique": row.nonunique == 0,
                        "column_names": [row.columnname],
                    }
        return [x[1] for x in indexes.items()]

    def do_executemany(self, cursor, statement, parameters, context=None):
        cursor.executemany(statement, parameters)

    def do_execute(self, cursor, statement, parameters, context=None):
        formatted_statement = ""
        for chunk in enumerate(statement.split("?")):
            if len(chunk[1]) > 0:
                formatted_statement += chunk[1]
            if chunk[0] < len(parameters):

                # If the parameter to be inserted is a string, enclose it in single quotes
                # If the parameter to be quotes already contains a single quote, quote that too
                if isinstance(parameters[chunk[0]], str):
                    safe = parameters[chunk[0]].strip().replace("'", "''")
                    formatted_statement += f"'{safe}'"

                # If the parameter to be inserted is some kind of number, add it as-is
                elif isinstance(parameters[chunk[0]], (int, float, Decimal,)):
                    formatted_statement += f"{parameters[chunk[0]]}"

                # The DF driver demands that dates be formatted as {d 'YYYY-MM-DD'}
                # and timestamps / datetimes as {ts 'YYYY-MM-DD HH:MM:SS.SSS'}
                elif isinstance(parameters[chunk[0]], date):
                    formatted_statement += (
                        "{d '" + f"{parameters[chunk[0]].isoformat()}" + "'}"
                    )
                elif isinstance(parameters[chunk[0]], datetime):
                    formatted_statement += (
                        "{ts '"
                        + f"{parameters[chunk[0]].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
                        + "'}"
                    )
                elif isinstance(parameters[chunk[0]], time):
                    raise NotImplementedError(
                        "DataFlex doesn't support time-only objects! Try using a date or datetime."
                    )

        # The DF driver doesn't support TOP or LIMIT constraints, but I'm not sure
        # where (other than here) is the best place to remove the generated TOP constraints
        #
        # This probably does technically break *something*, but in practice it hasn't affected
        # anything that wouldn't have thrown an error due to something else anyway
        formatted_statement = sub(r"SELECT TOP \d* ", r"SELECT ", formatted_statement)

        # Unquote this if you need to see the actual formatted query passed to the DF driver
        # Useful for testing / bug hunting / query fixing
        # print(f"Formatted Statement:\n{formatted_statement}\n")

        cursor.execute(formatted_statement)
