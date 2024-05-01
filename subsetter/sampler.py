import abc
import functools
import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import sqlalchemy as sa
from pydantic import BaseModel, Field
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ClauseElement, Executable
from typing_extensions import Annotated

from subsetter.common import DatabaseConfig, parse_table_name
from subsetter.filters import FilterConfig, FilterOmit, FilterView, FilterViewChain
from subsetter.metadata import DatabaseMetadata
from subsetter.plan_model import SQLTableIdentifier
from subsetter.planner import SubsetPlan
from subsetter.solver import toposort

try:
    from tqdm import tqdm  # type: ignore
except ImportError:

    def tqdm(x, **_):
        return x


LOGGER = logging.getLogger(__name__)
SOURCE_BUFFER_SIZE = 1024
DESTINATION_BUFFER_SIZE = 1024


# pylint: disable=too-many-ancestors,abstract-method
class TemporaryTable(Executable, ClauseElement):
    inherit_cache = True

    TEMP_ID = "_tmp_subsetter_"

    def __init__(
        self, schema: str, name: str, select: sa.Select, index: int = 0
    ) -> None:
        self.schema = schema
        self.name = name
        self.select = select
        self.index = index
        self.table_obj: sa.Table

    def _temporary_table_compile_generic(self, compiler, **_) -> str:
        name = self.name + self.TEMP_ID + str(self.index)
        schema_enc = compiler.dialect.identifier_preparer.quote(self.schema)
        name_enc = compiler.dialect.identifier_preparer.quote(name)
        select_stmt = compiler.process(self.select)
        self.table_obj = sa.Table(
            name,
            sa.MetaData(),
            *(sa.Column(col.name, col.type) for col in self.select.selected_columns),
            schema=self.schema,
        )
        return f"CREATE TEMPORARY TABLE {schema_enc}.{name_enc} AS {select_stmt}"

    def _temporary_table_compile_postgres(self, compiler, **_) -> str:
        """
        Postgres creates temporary tables in a special schema. We make the table
        name incorporate the schema name to compensate and avoid collisions.
        """
        name = self.schema + self.TEMP_ID + f"{self.index}_" + self.name
        name_enc = compiler.dialect.identifier_preparer.quote(name)
        select_stmt = compiler.process(self.select)
        self.table_obj = sa.Table(
            name,
            sa.MetaData(),
            *(sa.Column(col.name, col.type) for col in self.select.selected_columns),
        )
        return f"CREATE TEMPORARY TABLE {name_enc} AS {select_stmt}"


compiles(TemporaryTable, "postgresql")(TemporaryTable._temporary_table_compile_postgres)
compiles(TemporaryTable)(TemporaryTable._temporary_table_compile_generic)


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


class TableRemapPattern(BaseModel):
    search: str
    replace: str


class DatabaseOutputConfig(DatabaseConfig):
    mode: Literal["database"]
    remap: List[TableRemapPattern] = []


OutputType = Annotated[
    Union[DirectoryOutputConfig, DatabaseOutputConfig],
    Field(..., discriminator="mode"),
]


class SamplerConfig(BaseModel):
    source: DatabaseConfig = DatabaseConfig()
    output: OutputType = DirectoryOutputConfig(mode="directory", directory="output")
    filters: Dict[str, List[FilterConfig]] = {}  # type: ignore
    multiplicity: MultiplicityConfig = MultiplicityConfig()


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

    def truncate(self) -> None:
        """Delete any existing data that could interfere with output destination"""

    @abc.abstractmethod
    def insert_order(self) -> List[str]:
        """Return the order to insert data that respects foreign key relationships"""

    @staticmethod
    def from_config(config: Any, tables: List[str]) -> "SamplerOutput":
        if isinstance(config, DirectoryOutputConfig):
            return DirectoryOutput(config, tables)
        if isinstance(config, DatabaseOutputConfig):
            return DatabaseOutput(config, tables)
        raise RuntimeError("Unknown config type")


class DirectoryOutput(SamplerOutput):
    def __init__(self, config: DirectoryOutputConfig, tables: List[str]) -> None:
        self.directory = config.directory
        self.tables = list(tables)

    def insert_order(self) -> List[str]:
        return self.tables

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


class DatabaseOutput(SamplerOutput):
    # Pylint getting weirdly confused about the self.meta member
    # pylint: disable=no-member

    def __init__(self, config: DatabaseOutputConfig, tables: List[str]) -> None:
        self.engine = config.database_engine(env_prefix="SUBSET_DESTINATION_")
        self.remap = config.remap

        self.table_remap: Dict[str, str] = {}
        for table in tables:
            remapped_table = ".".join(self._remap_table(*parse_table_name(table)))
            if self.table_remap.setdefault(remapped_table, table) != table:
                raise ValueError(
                    f"Multiple tables remapped to the same name {remapped_table}"
                )

        self.meta, self.additional_tables = DatabaseMetadata.from_engine(
            self.engine,
            list(self.table_remap),
            close_backward=True,
        )

    def truncate(self) -> None:
        for schema, table_name in self.additional_tables:
            LOGGER.info(
                "Found additional table %s.%s to truncate",
                schema,
                table_name,
            )

        with self.engine.connect() as conn:
            for table in reversed(self.meta.toposort()):
                LOGGER.info("Truncating table %s", table)
                conn.execute(sa.delete(table.table_obj))
                conn.commit()

    def _remap_table(self, schema: str, table_name: str) -> Tuple[str, str]:
        table = f"{schema}.{table_name}"
        for remap_pattern in self.remap:
            table = re.sub(remap_pattern.search, remap_pattern.replace, table)
        return parse_table_name(table)

    def insert_order(self) -> List[str]:
        result = []
        insert_order_mapped = [str(table) for table in self.meta.toposort()]
        for table in insert_order_mapped:
            remapped_table = self.table_remap.get(table)
            if remapped_table:
                result.append(remapped_table)
            else:
                LOGGER.warning(
                    "Database does not have table %s, will not sample", table
                )
        return result

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
        schema, table_name = self._remap_table(schema, table_name)
        table = self.meta.tables[(schema, table_name)]

        # Automatically omit any included computed columns
        columns_out = filter_view.columns_out if filter_view else columns
        computed_columns = [
            col for col in columns_out if table.table_obj.columns[col].computed
        ]
        if computed_columns:
            omit_filter = FilterOmit(columns_out, computed_columns)
            columns_out = omit_filter.columns_out
            if filter_view:
                filter_view = FilterViewChain(
                    filter_view.columns_in,
                    columns_out,
                    (filter_view, omit_filter),
                )
            else:
                filter_view = omit_filter

        multiplied_indexes = [
            ind
            for ind, col_name in enumerate(columns_out)
            if col_name in (column_multipliers or set())
        ]

        buffer: List[tuple] = []

        def _flush_buffer():
            with self.engine.connect() as conn:
                conn.execute(
                    sa.insert(table.table_obj),
                    [dict(zip(columns_out, row)) for row in buffer],
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

    def sample(self, plan: SubsetPlan, *, truncate: bool = False) -> None:
        meta, _ = DatabaseMetadata.from_engine(self.source_engine, list(plan.queries))
        if self.config.multiplicity.infer_foreign_keys:
            meta.infer_missing_foreign_keys()
        self._validate_filters(meta)

        table_column_multipliers = self._get_multiplied_columns(
            meta, list(plan.queries)
        )

        output = SamplerOutput.from_config(self.config.output, list(plan.queries))
        insert_order = output.insert_order()
        if truncate:
            output.truncate()

        with self.source_engine.execution_options().connect() as conn:
            self._materialize_tables(meta, conn, plan)
            self._copy_results(
                output, conn, meta, plan, insert_order, table_column_multipliers
            )

    def _materialization_order(
        self, meta: DatabaseMetadata, plan: SubsetPlan
    ) -> List[Tuple[str, str, int]]:
        """
        Returns a list of tables that need to materialized. This list is a tuple
        of the form (schema, table, max_ref_count). Some dialects (i.e. mysql) require
        making multiple copies of a temp table if they are referenced multiple times.
        """

        def _record_sampled_tables(
            counter: Dict[Tuple[str, str], int], ident: SQLTableIdentifier
        ) -> sa.Table:
            key = (ident.table_schema, ident.table_name)
            if ident.sampled:
                counter[key] = counter.get(key, 0) + 1
            return meta.tables[key].table_obj

        dep_graph: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        max_ref_counts: Dict[Tuple[str, str], int] = {}
        for table, query in plan.queries.items():
            counter: Dict[Tuple[str, str], int] = {}
            query.build(functools.partial(_record_sampled_tables, counter))
            for key, count in counter.items():
                max_ref_counts[key] = max(max_ref_counts.get(key, 0), count)
            dep_graph[parse_table_name(table)] = set(counter.keys())

        order: List[Tuple[str, str]] = toposort(dep_graph)
        return [
            (schema, table_name, max_ref_counts[(schema, table_name)])
            for schema, table_name in order
            if (schema, table_name) in max_ref_counts
        ]

    def _materialize_tables(
        self,
        meta: DatabaseMetadata,
        conn: sa.Connection,
        plan: SubsetPlan,
    ) -> None:
        materialization_order = self._materialization_order(meta, plan)
        for schema, table_name, ref_count in materialization_order:
            query = plan.queries[f"{schema}.{table_name}"]
            ttbl = TemporaryTable(
                schema, table_name, query.build(meta.sql_build_context())
            )

            LOGGER.info(
                "Materializing sample for %s.%s",
                schema,
                table_name,
            )
            LOGGER.debug(
                "  Using statement %s",
                str(ttbl.compile(dialect=conn.engine.dialect)).replace("\n", " "),
            )

            try:
                result = conn.execute(ttbl)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Some client/server combinations report a read-only error even though the temporary
                # table creation actually succeeded. We'll just swallow the error here and if there
                # was a real issue it'll get flagged again when we query against it.
                if "--read-only" not in str(exc):
                    raise
            else:
                meta.temp_tables[(schema, table_name, 0)] = ttbl.table_obj
                LOGGER.info(
                    "Materialized %d rows for %s.%s in temporary table",
                    result.rowcount,
                    schema,
                    table_name,
                )

            if meta.supports_temp_reopen:
                continue

            # Create additional copies of the temporary table if needed. This is
            # to work around an issue on mysql with reopening temporary tables.
            for index in range(1, ref_count):
                ttbl_copy = TemporaryTable(
                    schema, table_name, sa.select(ttbl.table_obj), index=index
                )
                try:
                    result = conn.execute(ttbl_copy)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    if "--read-only" not in str(exc):
                        raise
                else:
                    meta.temp_tables[(schema, table_name, index)] = ttbl_copy.table_obj
                    LOGGER.info(
                        "Copied materialization of %s.%s",
                        schema,
                        table_name,
                    )

    def _copy_results(
        self,
        output: SamplerOutput,
        conn,
        meta: DatabaseMetadata,
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

            if (schema, table_name, 0) in meta.temp_tables:
                query_stmt = sa.select(meta.temp_tables[(schema, table_name, 0)])
            else:
                query_stmt = query.build(meta.sql_build_context())

            LOGGER.debug(
                "  Using statement %s",
                str(query_stmt.compile(dialect=conn.engine.dialect)).replace("\n", " "),
            )
            result = conn.execution_options(
                stream_results=True,
                yield_per=SOURCE_BUFFER_SIZE,
            ).execute(query_stmt)
            columns = result.keys()

            filter_view = FilterViewChain.construct_filter(
                columns,
                self.config.filters.get(table, []),
            )

            rows = 0

            def _count_rows(result):
                nonlocal rows
                for row in tqdm(result, desc="row progress", unit="rows"):
                    # result_processor
                    rows += 1
                    yield row

            column_multipliers = table_column_multipliers.get(table)
            output.output_result_set(
                schema,
                table_name,
                columns,
                _count_rows(result),
                filter_view=filter_view,
                multiplier=(
                    1
                    if column_multipliers is None
                    else self.config.multiplicity.multiplier
                ),
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

            FilterViewChain.construct_filter(
                tuple(column.name for column in tbl.table_obj.columns),
                filters,
            )

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
            col_map = {column.name: column for column in table.table_obj.columns}

            # Verify multiplied columns are integer
            for col_name in result.get(table_name, set()):
                col = col_map[col_name]
                if not issubclass(col.type.python_type, int):  # type: ignore
                    raise ValueError(
                        f"Primary key column {table_name}.{col_name} "
                        "must be integral when using multiplicity"
                    )

        return result
