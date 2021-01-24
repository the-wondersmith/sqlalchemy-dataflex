"""Setup file for SQLAlchemy-dataflex."""
# coding=utf-8

import os
import re

from setuptools import setup, find_packages

v = open(os.path.join(os.path.dirname(__file__), "sqlalchemy_dataflex", "__init__.py"))
VERSION = re.compile(r'.*__version__ = "(.*?)"', re.S).match(v.read()).group(1)
v.close()

readme = os.path.join(os.path.dirname(__file__), "README.rst")

setup(
    name="sqlalchemy-dataflex",
    version=VERSION,
    description="SQLAlchemy support for Dataflex flat-files",
    long_description="",
    url="https://github.com/the-wondersmith/sqlalchemy-dataflex",
    author="Mark S.",
    author_email="developers@pawn-pay.com",
    license="AGPL v3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Database :: Front-Ends",
        "Operating System :: OS Independent",
    ],
    keywords="SQLAlchemy DataFlex flat files",
    project_urls={
        "Documentation": "https://github.com/the-wondersmith/sqlalchemy-dataflex/wiki",
        "Source": "https://github.com/the-wondersmith/sqlalchemy-dataflex",
        "Tracker": "https://github.com/the-wondersmith/sqlalchemy-dataflex/issues",
    },
    packages=find_packages(include=["sqlalchemy_dataflex"]),
    include_package_data=True,
    install_requires=["SQLAlchemy", "pyodbc>=4.0.27", "python-dateutil"],
    zip_safe=False,
    entry_points={"sqlalchemy.dialects": ["dataflex.pyodbc = sqlalchemy_dataflex.pyodbc:DataflexDialect_pyodbc"]},
)
