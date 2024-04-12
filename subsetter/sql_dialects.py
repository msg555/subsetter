import abc
import json
from typing import Any, Iterable, List, Literal

import sqlalchemy as sa

DatabaseDialect = Literal[
    "mysql",
    "postgres",
]


class SQLDialectEncoder(abc.ABC):
    @staticmethod
    def from_dialect(dialect: DatabaseDialect) -> "SQLDialectEncoder":
        if dialect == "mysql":
            return MysqlDialectEncoder()
        if dialect == "postgres":
            return PostgresDialectEncoder()
        raise ValueError(f"Unknown dialect {dialect!r}")

    @abc.abstractmethod
    def identifier(self, identifier: str) -> str:
        """Encode an identifier"""

    @abc.abstractmethod
    def table_name(self, schema: str, table: str, *, sampled: bool = False) -> str:
        """Encode a table name"""

    @abc.abstractmethod
    def column_list(self, columns: Iterable[str]) -> str:
        """Encode a list of column names"""

    @abc.abstractmethod
    def make_temp_table(self, schema: str, table: str, as_sql: str) -> str:
        """Encode a statement to create a temporary table"""

    @abc.abstractmethod
    def random(self) -> str:
        """Encode a call to a random number function"""

    def result_processor(self, column_types, result) -> Iterable[List[Any]]:
        """Decode columns from result set"""
        # pylint: disable=unused-argument
        yield from result


class MysqlDialectEncoder(SQLDialectEncoder):
    def identifier(self, identifier: str) -> str:
        return f"`{identifier}`"

    def table_name(self, schema: str, table: str, *, sampled: bool = False) -> str:
        if not sampled:
            return f"{self.identifier(schema)}.{self.identifier(table)}"

        tmp_id = "_tmp_subsetter_"
        return f"{self.identifier(schema)}.{self.identifier(tmp_id + table)}"

    def column_list(self, columns: Iterable[str]) -> str:
        return ", ".join(self.identifier(column) for column in columns)

    def make_temp_table(self, schema: str, table: str, as_sql: str) -> str:
        return f"CREATE TEMPORARY TABLE {self.table_name(schema, table, sampled=True)} {as_sql}"

    def random(self) -> str:
        return "rand()"

    def result_processor(self, column_types, result) -> Iterable[List[Any]]:
        """Decode columns from result set"""
        processors = []
        for index, column_type in enumerate(column_types):
            if isinstance(column_type, sa.JSON):
                processors.append((index, lambda x: json.loads(x) if x else None))

        if not processors:
            yield from result
            return

        for row in result:
            nrow = list(row)
            for index, processor in processors:
                nrow[index] = processor(nrow[index])
            yield nrow


class PostgresDialectEncoder(SQLDialectEncoder):
    def identifier(self, identifier: str) -> str:
        return f'"{identifier}"'

    def table_name(self, schema: str, table: str, *, sampled: bool = False) -> str:
        if not sampled:
            return f"{self.identifier(schema)}.{self.identifier(table)}"

        tmp_id = "_tmp_subsetter_"
        return f"{self.identifier(schema + tmp_id + table)}"

    def column_list(self, columns: Iterable[str]) -> str:
        return ", ".join(self.identifier(column) for column in columns)

    def make_temp_table(self, schema: str, table: str, as_sql: str) -> str:
        return f"CREATE TEMPORARY TABLE {self.table_name(schema, table, sampled=True)} AS {as_sql}"

    def random(self) -> str:
        return "random()"
