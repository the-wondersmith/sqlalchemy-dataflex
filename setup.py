import os
import re

from setuptools import setup, find_packages

v = open(
    os.path.join(os.path.dirname(__file__), "sqlalchemy_dataflex", "__init__.py")
)
VERSION = re.compile(r'.*__version__ = "(.*?)"', re.S).match(v.read()).group(1)
v.close()

readme = os.path.join(os.path.dirname(__file__), "README.md")


setup(
    name="sqlalchemy-dataflex",
    version=VERSION,
    description="DataFlex support for SQLAlchemy",
    long_description=open(readme).read(),
    url="https://github.com/the-wondersmith/sqlalchemy-dataflex",
    author="Mark S.",
    author_email="developers@pawn-pay.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Database :: Front-Ends",
        "Operating System :: OS Independent",
    ],
    keywords=["SQLAlchemy", "DataFlex", "FlexODBC"],
    project_urls={
        "Source": "https://github.com/the-wondersmith/sqlalchemy-dataflex",
    },
    packages=find_packages(include=["sqlalchemy_dataflex"]),
    include_package_data=True,
    install_requires=["SQLAlchemy", "pyodbc>=4.0.27"],
    zip_safe=False,
    entry_points={
        "sqlalchemy.dialects": [
            "dataflex.pyodbc = sqlalchemy_dataflex.pyodbc:DataFlexDialect_pyodbc",
        ]
    },
)
