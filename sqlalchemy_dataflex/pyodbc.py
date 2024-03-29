"""SQLAlchemy Support for the Dataflex flat-file databases via pyodbc."""
# coding=utf-8


from .base import DataflexDialect, strtobool, cl_in, cg
from sqlalchemy.connectors.pyodbc import PyODBCConnector
from urllib.parse import unquote_plus
from sqlalchemy.engine.url import URL
from itertools import chain
from pathlib import Path
from typing import Any, List, Dict, Tuple, Union, Optional


# noinspection PyUnresolvedReferences
class DataflexDialect_pyodbc(PyODBCConnector, DataflexDialect):
    """A subclass of the DataflexDialect for pyodbc."""

    pyodbc_driver_name = "DataFlex Driver"

    v3_args = {
        "DRV": {
            "long_name": "Driver",
            "description": """
            Supplied directly to pyodbc as the `driver` argument.
            """,
            "valid_values": [],
        },
        "AC": {
            "long_name": "autocommit",
            "description": """
            Supplied directly to pyodbc as the `autocommit` argument.
            """,
            "valid_values": [True, False],
        },
        "DataPath": {
            "long_name": "DBQ",
            "description": """
            Absolute path to the directory containing the Dataflex flat-files and
            associated Filelist.cfg file.
            """,
            "valid_values": [],
        },
        "CFG": {
            "long_name": "CollateCFGPath",
            "description": """
            """,
            "valid_values": [],
        },
        "RO": {
            "long_name": "ReadOnly",
            "description": """
            """,
            "valid_values": ["Y", "N"],
        },
        "YD": {
            "long_name": "YearDigits",
            "description": """
            """,
            "valid_values": [2, 4],
        },
        "DO": {
            "long_name": "DecimalOption",
            "description": """
            """,
            "valid_values": [",", ".", None],
        },
        "RN": {
            "long_name": "DisplayRecnum",
            "description": """
            """,
            "valid_values": ["Y", "N"],
        },
        "UID": {
            "long_name": "Username",
            "description": """
            """,
            "valid_values": [],
        },
        "PWD": {
            "long_name": "Password",
            "description": """
            """,
            "valid_values": [],
        },
        "DLB": {
            "long_name": "DisplayLoginBox",
            "description": """
            """,
            "valid_values": ["Y", "N"],
        },
        "DSN": {
            "long_name": "DataSourceName",
            "description": """
            """,
            "valid_values": [],
        },
    }

    v4_args = {
        "UST": {
            "long_name": "UseSimulatedTransactions",
            "description": """
                """,
            "valid_values": ["Y", "N"],
        },
        "LVC": {
            "long_name": "ConvertToLongVARCHAR",
            "description": """
                """,
            "valid_values": ["Y", "N"],
        },
        "ESN": {
            "long_name": "ReturnEmptyStringsAsNULLs",
            "description": """
                """,
            "valid_values": ["Y", "N"],
        },
        "OC": {
            "long_name": "UseODBCCompatibility",
            "description": """
                """,
            "valid_values": ["Y", "N"],
        },
    }

    # noinspection Mypy
    @property
    def arg_name_map(self) -> Dict[str, Optional[Union[str, int]]]:
        """Mapping for long names to short names."""
        flexodbc_args = dict(chain(self.v3_args.items(), self.v4_args.items()))  # type: ignore
        return {flexodbc_args.get(key).get("long_name"): key for key in flexodbc_args}  # type: ignore

    def create_connect_args(self, url: URL) -> Tuple[List[Any], Dict[str, Optional[Union[int, str]]]]:
        """Create connection arguments from the supplied URL."""

        conn_args = {
            "autocommit": True,
            "Driver": "{DataFlex Driver}",
            "DataPath": "C:\\DataFlexData",
            "UseSimulatedTransactions": "Y",
            "ConvertToLongVARCHAR": "Y",
            "ReturnEmptyStringsAsNULLs": "N",
        }

        opts = url.translate_connect_args()

        if not opts:
            opts = dict()
            opts["host"] = url.host or url.query.get("odbc_connect", "")

        if cg(opts, "host", False):
            supplied_args = {
                pair[0]: pair[1]
                for pair in map(
                    lambda entry: entry.split("="),
                    chain.from_iterable(
                        map(
                            lambda item: item.split(";"),
                            unquote_plus(opts.get("host")).replace("?odbc_connect=", "").split("&"),
                        )
                    ),
                )
            }

            supplied_args = {
                value: cg(supplied_args, key) if not cl_in(value, supplied_args.keys()) else cg(supplied_args, value)
                for key, value in self.arg_name_map.items()
            }

            conn_args.update(
                {
                    "DSN": cg(supplied_args, "dsn"),
                    "autocommit": strtobool(cg(conn_args, "autocommit", cg(conn_args, "ac", True))),
                    "DBQ": cg(supplied_args, "DataPath", cg(supplied_args, "DBQ", conn_args.get("DataPath")))
                }
            )

        ret_val = (
            [],
            {
                key: value
                for key, value in conn_args.items()
                if all(
                    (
                        any((cl_in(key, self.arg_name_map.keys()), cl_in(key, self.arg_name_map.values()))),
                        value is not None,
                    )
                )
            },
        )

        return ret_val

    def connect(self, *args: Any, **kwargs: Any):
        """Establish a connection using pyodbc."""

        driver = cg(kwargs, "driver", "{DataFlex Driver}")
        autocommit = cg(kwargs, "autocommit", True)

        if cg(kwargs, "dsn", cg(kwargs, "DataSourceName", False)):
            return self.dbapi.connect(
                f"DSN={cg(kwargs, 'dsn', cg(kwargs, 'DataSourceName', ''))}",
                autocommit=autocommit,
            )

        conn_string = ";".join(
            (
                f"Driver={driver}",
                f"DBQ={cg(kwargs, 'dbq', cg(kwargs, 'DataPath', ''))}",
                ";".join(
                    (
                        f"{key}={value}"
                        for key, value in kwargs.items()
                        if all(
                            (
                                not cl_in(key, ("driver", "autocommit", "dsn", "dbq")),
                                cl_in(key, self.arg_name_map.keys()),
                            )
                        )
                    )
                ),
            )
        )

        return self.dbapi.connect(conn_string, autocommit=autocommit)
