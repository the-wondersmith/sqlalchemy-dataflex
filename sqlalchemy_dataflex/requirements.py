from sqlalchemy.testing.requirements import SuiteRequirements

from sqlalchemy.testing import exclusions


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
        """Target database allows boolean columns to store NULL."""
        # Access Yes/No doesn't allow null
        return exclusions.closed()

    @property
    def offset(self):
        # Access does LIMIT (via TOP) but not OFFSET
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
        # And even if it did, DataFlex won't let you drop a child table unless
        # you drop the FK constraint first. Not worth the grief.
        return exclusions.closed()

    @property
    def view_column_reflection(self):
        return exclusions.open()
