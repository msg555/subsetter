import abc
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Set, Union

import sqlalchemy as sa
from pydantic import BaseModel, Field
from pydantic.typing import Annotated

from subsetter.common import (
    DatabaseConfig,
    mysql_column_list,
    mysql_table_name,
    parse_table_name,
)
from subsetter.filters import FilterConfig, FilterView, FilterViewChain
from subsetter.metadata import DatabaseMetadata
from subsetter.planner import SubsetPlan

try:
    from tqdm import tqdm  # type: ignore
except ImportError:

    def tqdm(x, **_):
        return x


LOGGER = logging.getLogger(__name__)

# Name suffix to give to our temporary tables
TMP_SUFFIX = "_tmp_subsetter_"

SOURCE_BUFFER_SIZE = 1024
DESTINATION_BUFFER_SIZE = 1024


def _mangle_query(query: str) -> str:
    return query.replace("<SAMPLED>", TMP_SUFFIX)


def _multiply_column(
    value: Optional[int], multiplier: int, iteration: int
) -> Optional[int]:
    if value is None:
        return None
    return value * multiplier + iteration


class DirectoryOutputConfig(BaseModel):
    mode: Literal["directory"]
    directory: str


class MultiplicityConfig(BaseModel):
    multiplier: int = 1
    infer_foreign_keys: bool = False
    passthrough: List[str] = []
    extra_columns: Dict[str, List[str]] = {}
    ignore_primary_key_columns: Dict[str, List[str]] = {}


class MysqlOutputConfig(DatabaseConfig):
    mode: Literal["mysql"]


OutputType = Annotated[
    Union[DirectoryOutputConfig, MysqlOutputConfig],
    Field(..., discriminator="mode"),
]


class SamplerConfig(BaseModel):
    source: DatabaseConfig = DatabaseConfig()
    output: OutputType = DirectoryOutputConfig(mode="directory", directory="output")
    filters: Dict[str, List[FilterConfig]] = {}  # type: ignore
    multiplicity: MultiplicityConfig = MultiplicityConfig()


SamplerConfig.update_forward_refs()


class SamplerOutput(abc.ABC):
    def output_result_set(
        self,
        schema: str,
        table_name: str,
        columns: List[str],
        result_set,
        *,
        filter_view: Optional[FilterView] = None,
        multiplier: int = 1,
        column_multipliers: Optional[Set[str]] = None,
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
        self,
        schema: str,
        table_name: str,
        columns: List[str],
        result_set,
        *,
        filter_view: Optional[FilterView] = None,
        multiplier: int = 1,
        column_multipliers: Optional[Set[str]] = None,
    ) -> None:
        output_path = os.path.join(self.directory, f"{schema}.{table_name}.json")
        columns_out = filter_view.columns_out if filter_view else columns

        multiplied_indexes = [
            ind
            for ind, col_name in enumerate(columns_out)
            if col_name in (column_multipliers or set())
        ]
        with open(output_path, "w", encoding="utf-8") as fdump:
            for row in result_set:
                for iteration in range(multiplier):
                    out_row = filter_view.filter_view(row) if filter_view else list(row)
                    for index in multiplied_indexes:
                        out_row[index] = _multiply_column(
                            out_row[index], multiplier, iteration
                        )
                    json.dump(dict(zip(columns_out, out_row)), fdump, default=str)
                    fdump.write("\n")


class MysqlOutput(SamplerOutput):
    def __init__(self, config: MysqlOutputConfig) -> None:
        self.engine = config.database_engine(env_prefix="SUBSET_DESTINATION_")

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
        self,
        schema: str,
        table_name: str,
        columns: List[str],
        result_set,
        *,
        filter_view: Optional[FilterView] = None,
        multiplier: int = 1,
        column_multipliers: Optional[Set[str]] = None,
    ) -> None:
        columns_out = filter_view.columns_out if filter_view else columns
        multiplied_indexes = [
            ind
            for ind, col_name in enumerate(columns_out)
            if col_name in (column_multipliers or set())
        ]
        insert_query = (
            f"INSERT INTO {mysql_table_name(schema, table_name)} "
            f"({mysql_column_list(columns_out)}) VALUES "
        )
        buffer: List[tuple] = []

        def _flush_buffer():
            query = f"{insert_query}{','.join(f':{ind}' for ind in range(len(buffer)))}"
            with self.engine.connect() as conn:
                conn.execute(
                    sa.text(query),
                    {str(ind): row for ind, row in enumerate(buffer)},
                )
                conn.commit()
            buffer.clear()

        for row in result_set:
            for iteration in range(multiplier):
                out_row = filter_view.filter_view(row) if filter_view else row
                for index in multiplied_indexes:
                    out_row[index] = _multiply_column(
                        out_row[index], multiplier, iteration
                    )
                buffer.append(tuple(out_row))
                if len(buffer) > DESTINATION_BUFFER_SIZE:
                    _flush_buffer()
        if buffer:
            _flush_buffer()


class Sampler:
    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.source_engine = self.config.source.database_engine(
            env_prefix="SUBSET_SOURCE_"
        )
        self.output = SamplerOutput.from_config(config.output)

    def sample(self, plan: SubsetPlan, *, truncate: bool = False) -> None:
        meta, _ = DatabaseMetadata.from_engine(self.source_engine, list(plan.queries))
        if self.config.multiplicity.infer_foreign_keys:
            meta.infer_missing_foreign_keys()
        self._validate_filters(meta)

        insert_order = self.output.insert_order(list(plan.queries), truncate=truncate)
        table_column_multipliers = self._get_multiplied_columns(
            meta, list(plan.queries)
        )

        if truncate:
            self._truncate(insert_order)
        with self.source_engine.execution_options(
            stream_results=True,
            yield_per=SOURCE_BUFFER_SIZE,
        ).connect() as conn:
            self._materialize_tables(conn, plan)
            self._copy_results(conn, plan, insert_order, table_column_multipliers)

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

    def _copy_results(
        self,
        conn,
        plan: SubsetPlan,
        insert_order: List[str],
        table_column_multipliers: Dict[str, Set[str]],
    ):
        for table in tqdm(insert_order, desc="table progress", unit="tables"):
            schema, table_name = parse_table_name(table)

            query = plan.queries.get(table)
            if query is None:
                continue

            LOGGER.info("Sampling %s.%s ...", schema, table_name)

            if query.materialize:
                result = conn.execution_options(yield_per=SOURCE_BUFFER_SIZE).execute(
                    sa.text(
                        f"SELECT * FROM {mysql_table_name(schema, table_name + TMP_SUFFIX)}"
                    )
                )
            else:
                result = conn.execution_options(yield_per=SOURCE_BUFFER_SIZE).execute(
                    sa.text(_mangle_query(query.query)),
                    query.params,
                )
            columns = result.keys()

            filter_view = FilterViewChain.construct_filter(
                columns,
                self.config.filters.get(table, []),
            )

            rows = 0

            def _count_rows(result):
                nonlocal rows
                for row in tqdm(result, desc="row progress", unit="rows"):
                    rows += 1
                    yield row

            column_multipliers = table_column_multipliers.get(table)
            self.output.output_result_set(
                schema,
                table_name,
                columns,
                _count_rows(result),
                filter_view=filter_view,
                multiplier=1
                if column_multipliers is None
                else self.config.multiplicity.multiplier,
                column_multipliers=column_multipliers,
            )
            LOGGER.info("Sampled %d rows for %s.%s", rows, schema, table_name)

    def _validate_filters(self, meta: DatabaseMetadata):
        for table, filters in self.config.filters.items():
            if not filters:
                continue

            schema, table_name = parse_table_name(table)
            tbl = meta.tables.get((schema, table_name))
            if tbl is None:
                LOGGER.warning("Found filters for unknown table %s", table)
                continue

            FilterViewChain.construct_filter(tbl.columns, filters)

    def _get_multiplied_columns(
        self, meta: DatabaseMetadata, tables: List[str]
    ) -> Dict[str, Set[str]]:
        """
        Computes a mapping of tables to the list of columns that should be multiplied.
        Here a 'multiplied' column must be an integer column that should be updated as
        (column_value * multiplier + i) for each multiplied record 0 <= i < multiplier.

        Generally any column that's not part of a passthrough table that meets these
        criteria will be a 'multiplied' column:
        - Is explicitly listed in multiplicity.extra_columns
        - Is part of a primary key and table not listed in multiplicity.ignore_primary_key
        - Is part of a foreign key to a non-passthrough table

        Additionally the current implementation requires all multiplied columns to be
        integral. This method will raise a ValueError if this is not the case.
        """
        if self.config.multiplicity.multiplier <= 1:
            return {}

        result: Dict[str, Set[str]] = {}
        passthrough_tables = set(self.config.multiplicity.passthrough)

        for table_name in tables:
            if table_name in passthrough_tables:
                continue

            table = meta.tables[parse_table_name(table_name)]

            # Calculate set of directly multiplied columns for this table
            cols = set(self.config.multiplicity.extra_columns.get(table_name, []))
            ignored_pk_cols = set(
                self.config.multiplicity.ignore_primary_key_columns.get(table_name, [])
            )
            cols.update(set(table.primary_key) - ignored_pk_cols)

            result[table_name] = cols

        # Multiply foreing key columns that point at multplied columns. Being lazy
        # about this closure; realistically most databases only have FKs that point directly
        # at PKs so this should loop twice.
        while True:
            changes = False

            for table_name in tables:
                table = meta.tables[parse_table_name(table_name)]

                cols = result.get(table_name, set())
                start_len = len(cols)
                for fk in table.foreign_keys:
                    dst_mapped = result.get(f"{fk.dst_schema}.{fk.dst_table}", set())
                    cols.update(set(fk.columns) & dst_mapped)

                if cols and table_name in passthrough_tables:
                    raise ValueError(
                        "Passthrough foreign key points to multiplied column"
                    )

                changes = changes or len(cols) > start_len

            if not changes:
                break

        for table_name in tables:
            table = meta.tables[parse_table_name(table_name)]

            # Verify multiplied columns are integer
            for col_name in result.get(table_name, set()):
                col = table.columns[col_name]
                if not issubclass(col.type_.python_type, int):  # type: ignore
                    raise ValueError(
                        f"Primary key column {table_name}.{col_name} "
                        "must be integral when using multiplicity"
                    )

        return result
