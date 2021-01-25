# SQLAlchemy-DataFlex

A SQLAlchemy dialect for the
[FlexODBC driver](https://www.flextools.com/flexodbc)

## Objectives

This dialect is mainly intended to offer an easy way to access the
DataFlex flat-file databases of older or EOL'd application-specific
softwares. It is designed for use with the ODBC driver available from
[FlexQuarters](http://flexquarters.com/)

## Pre-requisites

- The [FlexODBC driver](https://www.flextools.com/flexodbc) from
  [FlexQuarters](http://flexquarters.com/). ~~The library was written and
  tested against v3, but it *should* work with v4 as well.~~ The library
  has been re-written from the ground up and tested to work against the
  latest available version of FlexODBC (4.0.27.2).

- 32-bit Python. Neither of the available FlexODBC driver versions
  support or come in a 64-bit version.

## Co-requisites

This dialect requires SQLAlchemy and pyodbc. They are both specified as
requirements so `pip` will install them if they are not already in
place. To install separately, just:

> `pip install -U SQLAlchemy pyodbc`

## Installation

PyPI-published version:

> `pip install -U sqlalchemy-dataflex`

Absolute bleeding-edge (probably buggy):

> `pip install -U git+https://github.com/the-wondersmith/sqlalchemy-dataflex`

## Getting Started

Create an `ODBC DSN (Data Source Name)` that points to the directory
containing your DataFlex `FILELIST.cfg` and `.DAT` table.

> Tip: For best results, be sure to select the 4-digit date option and
> the `.` option for decimals

Then, in your Python app, you can connect to the database via:

```python
from sqlalchemy_dataflex import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


db = create_engine("dataflex+pyodbc://@your_dsn", echo=False)
super_session = sessionmaker(bind=db)
super_session.configure(autoflush=True, autocommit=True, expire_on_commit=True)
session = super_session()
```

> NOTE: It is *highly* recommended that you use the datatype classes from the package itself
>       in place of the ones that are usually imported from SQLAlchemy. The testing suite doesn't
>       use the SQLAlchemy type classes, so any issues that might arise from their use with the dialect
>       won't be caught by the tests.

## The SQLAlchemy Project

SQLAlchemy-DataFlex is based on SQLAlchemy-access, which is part of the
[SQLAlchemy Project] (https://www.sqlalchemy.org) and *tries* to adhere
to the same standards and conventions as the core project.

At the time of this writing, it's unlikely that SQLAlchemy-DataFlex
actually *does* comply with the aforementioned standards and
conventions. That will be rectified (if and when) in a future release.

## Development / Bug reporting / Pull requests

SQLAlchemy maintains a
[Community Guide](https://www.sqlalchemy.org/develop.html) detailing
guidelines on coding and participating in that project.

While I'm aware that this project could desperately use the
participation of anyone else who actually knows what they're doing,
DataFlex and the FlexODBC driver are so esoteric and obscure (at the
time of this writing) that I don't reasonably expect anyone to actually
want to. If I am mistaken in that belief, *please God* submit a pull
request.

This library technically *works*, but it isn't feature-complete so to
speak. The FlexODBC driver only supports a very limited subset of SQL
commands and queries, and doesn't always respond to pyodbc's
`get_info()` inquiries the way that pyodbc is expecting. You can see
the complete list of the way it responds to all of the available pyodbc
`get_info()` inquiries [here](./flexodbc_capabilities.json). If you are
interested in which features are currently lacking or absent in the dialect,
see the notes and comments littered throughout the testing suite [here](./tests/test_suite.py).

## License

This library is derived almost in its entirety from the
SQLAlchemy-Access library written by
[Gord Thompson](https://github.com/gordthompson). As such, and given
that SQLAlchemy-access is distributed under the
[MIT license](https://opensource.org/licenses/MIT), this library is
subject to the same licensure and grant of rights as its parent works
[SQLAlchemy](https://www.sqlalchemy.org/) and
[SQLAlchemy-Access](https://github.com/sqlalchemy/sqlalchemy-access).
