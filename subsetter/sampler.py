import abc
import json
import logging
import os
from typing import Annotated, Any, List, Literal, Union

import sqlalchemy as sa
from pydantic import BaseModel, Field

from subsetter.common import (
    DatabaseConfig,
    database_url,
    mysql_column_list,
    mysql_table_name,
    parse_table_name,
)
from subsetter.metadata import DatabaseMetadata
from subsetter.planner import SubsetPlan

LOGGER = logging.getLogger(__name__)

# Name suffix to give to our temporary tables
TMP_SUFFIX = "_tmp_subsetter_"


def _mangle_query(query: str) -> str:
    return query.replace("<SAMPLED>", TMP_SUFFIX)


class DirectoryOutputConfig(BaseModel):
    mode: Literal["directory"]
    directory: str


class MysqlOutputConfig(DatabaseConfig):
    mode: Literal["mysql"]


OutputType = Annotated[
    Union[DirectoryOutputConfig, MysqlOutputConfig],
    Field(..., discriminator="mode"),
]


class SamplerConfig(BaseModel):
    source: DatabaseConfig = DatabaseConfig()
    output: OutputType = DirectoryOutputConfig(mode="directory", directory="output")


class SamplerOutput(abc.ABC):
    def output_result_set(
        self, schema: str, table_name: str, columns: List[str], result_set
    ) -> None:
        pass

    def truncate(self, schema: str, table_name: str) -> None:
        pass

    def insert_order(self, tables: List[str], *, truncate: bool = False) -> List[str]:
        # pylint: disable=unused-argument
        return tables

    @staticmethod
    def from_config(config: Any) -> "SamplerOutput":
        if isinstance(config, DirectoryOutputConfig):
            return DirectoryOutput(config)
        if isinstance(config, MysqlOutputConfig):
            return MysqlOutput(config)
        raise RuntimeError("Unknown config type")


class DirectoryOutput(SamplerOutput):
    def __init__(self, config: DirectoryOutputConfig) -> None:
        self.directory = config.directory

    def output_result_set(
        self, schema: str, table_name: str, columns: List[str], result_set
    ) -> None:
        output_path = os.path.join(self.directory, f"{schema}.{table_name}.json")
        with open(output_path, "w", encoding="utf-8") as fdump:
            for row in result_set:
                json.dump(dict(zip(columns, row)), fdump, default=str)
                fdump.write("\n")


class MysqlOutput(SamplerOutput):
    def __init__(self, config: MysqlOutputConfig) -> None:
        self.engine = sa.create_engine(
            database_url(
                env_prefix="SUBSET_DESTINATION_",
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
            )
        )

    def truncate(self, schema: str, table_name: str) -> None:
        LOGGER.info("Truncating table %s.%s", schema, table_name)
        with self.engine.connect() as conn:
            conn.execute(
                sa.text(f"DELETE FROM {mysql_table_name(schema, table_name)} WHERE 1")
            )
            conn.commit()

    def insert_order(self, tables: List[str], *, truncate: bool = True) -> List[str]:
        meta, additional_tables = DatabaseMetadata.from_engine(
            self.engine,
            tables,
            close_backward=truncate,
        )
        for table in tables:
            if parse_table_name(table) not in meta.tables:
                LOGGER.warning(
                    "Database does not have table %s, will not sample", table
                )
        for schema, table_name in additional_tables:
            LOGGER.info(
                "Found additional unsampled table %s.%s to truncate",
                schema,
                table_name,
            )
        return [str(table) for table in meta.toposort()]

    def output_result_set(
        self, schema: str, table_name: str, columns: List[str], result_set
    ) -> None:
        insert_query = (
            f"INSERT INTO {mysql_table_name(schema, table_name)} "
            f"({mysql_column_list(columns)}) VALUES "
        )
        with self.engine.connect() as conn:
            flush_count = 1024
            buffer: List[tuple] = []

            def _flush_buffer():
                query = (
                    f"{insert_query}{','.join(f':{ind}' for ind in range(len(buffer)))}"
                )
                conn.execute(
                    sa.text(query),
                    {str(ind): row for ind, row in enumerate(buffer)},
                )
                buffer.clear()

            for row in result_set:
                buffer.append(tuple(row))
                if len(buffer) > flush_count:
                    _flush_buffer()
            if buffer:
                _flush_buffer()

            conn.commit()


class Sampler:
    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.source_engine = sa.create_engine(
            database_url(
                env_prefix="SUBSET_SOURCE_",
                host=self.config.source.host,
                port=self.config.source.port,
                username=self.config.source.username,
                password=self.config.source.password,
            )
        )
        self.output = SamplerOutput.from_config(config.output)

    def sample(self, plan: SubsetPlan, *, truncate: bool = False) -> None:
        insert_order = self.output.insert_order(list(plan.queries), truncate=truncate)

        if truncate:
            self._truncate(insert_order)
        with self.source_engine.connect() as conn:
            self._materialize_tables(conn, plan)
            self._copy_results(conn, plan, insert_order)

    def _truncate(self, insert_order: List[str]) -> None:
        for table in reversed(insert_order):
            schema, table_name = parse_table_name(table)
            self.output.truncate(schema, table_name)

    def _materialize_tables(self, conn, plan: SubsetPlan) -> None:
        for table, query in plan.queries.items():
            if not query.materialize:
                continue
            schema, table_name = parse_table_name(table)

            LOGGER.info("Materializing sample for %s", table)
            try:
                result = conn.execute(
                    sa.text(
                        f"CREATE TEMPORARY TABLE {mysql_table_name(schema, table_name + TMP_SUFFIX)} "
                        + _mangle_query(query.query)
                    ),
                    query.params,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Some client/server combinations report a read-only error even though the temporary
                # table creation actually succeeded. We'll just swallow the error here and if there
                # was a real issue it'll get flagged again when we query against it.
                if "--read-only" not in str(exc):
                    raise
            else:
                LOGGER.info(
                    "Materialized %d rows for %s.%s in temporary table",
                    result.rowcount,
                    schema,
                    table_name,
                )

    def _copy_results(self, conn, plan: SubsetPlan, insert_order: List[str]):
        for table in insert_order:
            schema, table_name = parse_table_name(table)

            query = plan.queries.get(table)
            if query is None:
                continue

            LOGGER.info("Sampling %s.%s ...", schema, table_name)

            if query.materialize:
                result = conn.execute(
                    sa.text(
                        f"SELECT * FROM {mysql_table_name(schema, table_name + TMP_SUFFIX)}"
                    )
                )
            else:
                result = conn.execute(
                    sa.text(_mangle_query(query.query)),
                    query.params,
                )
            columns = result.keys()

            rows = 0

            def _count_rows(result):
                nonlocal rows
                for row in result:
                    rows += 1
                    yield row

            self.output.output_result_set(
                schema,
                table_name,
                columns,
                _count_rows(result),
            )
            LOGGER.info("Sampled %d rows for %s.%s", rows, schema, table_name)
