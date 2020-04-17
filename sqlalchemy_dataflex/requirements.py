from sqlalchemy.testing.requirements import SuiteRequirements

from sqlalchemy.testing import exclusions


# This entire file is more or less unaltered from the original SQLAlchemy-Access version
# Only a few notes are added


class Requirements(SuiteRequirements):
    @property
    def bound_limit_offset(self):
        return exclusions.closed()

    @property
    def date(self):
        return exclusions.closed()

    @property
    def datetime_microseconds(self):
        return exclusions.closed()

    @property
    def floats_to_four_decimals(self):
        return exclusions.closed()

    @property
    def foreign_key_constraint_reflection(self):
        return exclusions.closed()

    @property
    def nullable_booleans(self):
        """Target database doesn't support boolean columns"""
        # DataFlex doesn't support booleans at all really
        return exclusions.closed()

    @property
    def offset(self):
        # ADataFlex doesn't support LIMIT, TOP, or OFFSET
        return exclusions.closed()

    @property
    def parens_in_union_contained_select_w_limit_offset(self):
        return exclusions.closed()

    @property
    def precision_generic_float_type(self):
        return exclusions.closed()

    @property
    def primary_key_constraint_reflection(self):
        return exclusions.closed()

    @property
    def sql_expression_limit_offset(self):
        return exclusions.closed()

    @property
    def temp_table_reflection(self):
        return exclusions.closed()

    @property
    def temporary_tables(self):
        return exclusions.closed()

    @property
    def temporary_views(self):
        return exclusions.closed()

    @property
    def time(self):
        return exclusions.closed()

    @property
    def time_microseconds(self):
        return exclusions.closed()

    @property
    def timestamp_microseconds(self):
        return exclusions.closed()

    @property
    def unicode_ddl(self):
        # DataFlex ODBC does not support `SQLForeignKeys` so test teardown code
        # cannot determine the correct order in which to drop the tables.
        # And even if it did, DataFlex won't really let you drop a table anyway
        return exclusions.closed()

    @property
    def view_column_reflection(self):
        return exclusions.open()
