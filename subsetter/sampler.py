import abc
import functools
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.expression import ClauseElement, Executable

from subsetter.common import DatabaseConfig, parse_table_name
from subsetter.config_model import (
    ConflictStrategy,
    DatabaseOutputConfig,
    DirectoryOutputConfig,
    SamplerConfig,
)
from subsetter.filters import FilterOmit, FilterView, FilterViewChain
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

    def _temporary_table_compile_generic(self, compiler: SQLCompiler, **_) -> str:
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

    def _temporary_table_compile_no_schema(self, compiler: SQLCompiler, **_) -> str:
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


compiles(TemporaryTable, "postgresql", "sqlite")(
    TemporaryTable._temporary_table_compile_no_schema
)
compiles(TemporaryTable)(TemporaryTable._temporary_table_compile_generic)


# pylint: disable=too-many-ancestors,abstract-method
class ConflictInsert(Executable, ClauseElement):
    inherit_cache = False
    _inline = False
    _return_defaults = False

    def __init__(
        self,
        conflict_strategy: ConflictStrategy,
        table: sa.Table,
        columns: Iterable[str],
    ) -> None:
        self.conflict_strategy = conflict_strategy
        self.table = table
        self.non_pk_columns = set(columns)

        if not self.table.primary_key:
            # We only attempt to do this if there is a primary key
            self.conflict_strategy = "error"
        else:
            for column in self.table.primary_key:
                self.non_pk_columns.remove(column.name)
            if self.conflict_strategy == "replace" and not self.non_pk_columns:
                # If there are no other columns outside of the primary key the replace
                # strategy is meaningless and we should just skip.
                self.conflict_strategy = "skip"

    def _insert_generic(self, compiler: SQLCompiler, **kwargs) -> str:
        if self.conflict_strategy == "replace":
            raise RuntimeError(
                "'replace' conflict strategy is not supported for this dialect"
            )
        if self.conflict_strategy == "skip":
            raise RuntimeError(
                "'skip' conflict strategy is not supported for this dialect"
            )
        return compiler.process(sa.insert(self.table), **kwargs)

    def _insert_mysql(self, compiler: SQLCompiler, **kwargs) -> str:
        stmt = mysql.insert(self.table)
        if self.conflict_strategy == "replace":
            stmt = stmt.on_duplicate_key_update(
                {col: stmt.inserted[col] for col in self.non_pk_columns}
            )
        if self.conflict_strategy == "skip":
            stmt = stmt.prefix_with("IGNORE")
        return compiler.process(stmt, **kwargs)

    def _insert_postgresql(self, compiler: SQLCompiler, **kwargs) -> str:
        stmt = postgresql.insert(self.table)
        if self.conflict_strategy == "replace":
            stmt = stmt.on_conflict_do_update(
                index_elements=self.table.primary_key,
                set_={col: stmt.excluded[col] for col in self.non_pk_columns},
            )
        if self.conflict_strategy == "skip":
            stmt = stmt.on_conflict_do_nothing()
        return compiler.process(stmt, **kwargs)

    def _insert_sqlite(self, compiler: SQLCompiler, **kwargs) -> str:
        stmt = sqlite.insert(self.table)
        if self.conflict_strategy == "replace":
            stmt = stmt.on_conflict_do_update(
                index_elements=self.table.primary_key,
                set_=stmt.excluded,  # type: ignore
            )
        if self.conflict_strategy == "skip":
            stmt = stmt.on_conflict_do_nothing()
        return compiler.process(stmt, **kwargs)


compiles(ConflictInsert, "sqlite")(ConflictInsert._insert_sqlite)
compiles(ConflictInsert, "postgresql")(ConflictInsert._insert_postgresql)
compiles(ConflictInsert, "mysql")(ConflictInsert._insert_mysql)
compiles(ConflictInsert)(ConflictInsert._insert_generic)


def _autoincrement_needs_updating(engine) -> bool:
    """
    Returns True if the engine doesn't automatically update auto-increment values
    to the highest inserted value for that column. Currently only postgresql
    does not do this among supported engines.
    """
    return engine.dialect.name == "postgresql"


def _autoincrement_get(engine, table: sa.Table) -> Optional[int]:
    """
    Get the current auto-increment state.
    """
    if not _autoincrement_needs_updating(engine) or table.autoincrement_column is None:
        return None
    assert engine.dialect.name == "postgresql"

    try:
        with engine.connect() as conn:
            result = conn.execute(
                sa.text("SELECT currval(pg_get_serial_sequence(:tab, :col))"),
                {
                    "tab": f"{table.schema}.{table.name}",
                    "col": table.autoincrement_column.name,
                },
            )
            row = next(iter(result), None)
            return row[0] if row else None
    except sa.exc.OperationalError as exc:
        if "not yet defined in this session" not in str(exc):
            raise
        return None


def _autoincrement_update(engine, table: sa.Table, value: Optional[int]) -> None:
    """
    Update the auto-increment state to ensure the next generated value is greater
    than value.
    """
    if (
        not _autoincrement_needs_updating(engine)
        or table.autoincrement_column is None
        or value is None
    ):
        return
    assert engine.dialect.name == "postgresql"

    curval = _autoincrement_get(engine, table)
    if curval is not None and curval > value:
        return

    with engine.connect() as conn:
        conn.execute(
            sa.text("SELECT setval(pg_get_serial_sequence(:tab, :col), :val)"),
            {
                "tab": f"{table.schema}.{table.name}",
                "col": table.autoincrement_column.name,
                "val": value,
            },
        )
        conn.commit()


def _multiply_column(
    value: Optional[int], multiplier: int, iteration: int
) -> Optional[int]:
    if value is None:
        return None
    return value * multiplier + iteration


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

    def create(self) -> None:
        """Create any missing tables in destination from the source schema"""

    def prepare(self) -> None:
        """Called just prior to sampling after truncate/create have executed"""

    @abc.abstractmethod
    def insert_order(self) -> List[str]:
        """Return the order to insert data that respects foreign key relationships"""

    @staticmethod
    def from_config(
        config: Any, plan: SubsetPlan, source_meta: DatabaseMetadata
    ) -> "SamplerOutput":
        if isinstance(config, DirectoryOutputConfig):
            return DirectoryOutput(config, plan, source_meta)
        if isinstance(config, DatabaseOutputConfig):
            return DatabaseOutput(config, plan, source_meta)
        raise RuntimeError("Unknown config type")


class DirectoryOutput(SamplerOutput):
    def __init__(
        self,
        config: DirectoryOutputConfig,
        plan: SubsetPlan,
        source_meta: DatabaseMetadata,
    ) -> None:
        self.directory = config.directory
        self.plan = plan
        self.source_meta = source_meta

    def insert_order(self) -> List[str]:
        return list(self.plan.queries)

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

    def __init__(
        self,
        config: DatabaseOutputConfig,
        plan: SubsetPlan,
        source_meta: DatabaseMetadata,
    ) -> None:
        self.engine = config.database_engine(env_prefix="SUBSET_DESTINATION_")
        self.remap = config.remap
        self.conflict_strategy = config.conflict_strategy
        self.merge = config.merge
        self.plan = plan
        self.source_meta = source_meta
        self.table_offsets: Dict[Tuple[str, str], Dict[str, int]] = {}
        self.passthrough_tables = set(self.plan.passthrough)

        self.table_remap: Dict[str, str] = {}
        for table in plan.queries:
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

    def prepare(self) -> None:
        """
        If merge mode is enabled calculate the offsets to assign to each primary
        key for each table. Uses the foreign key analysis done at the source
        database to inform what foreign key relationships are expected to exist
        in the destination.
        """
        if not self.merge:
            return

        pk_max_values = {}
        self.table_offsets.clear()
        for source_table in self.plan.queries:
            if source_table in self.passthrough_tables:
                continue

            schema, table_name = self._remap_table(*parse_table_name(source_table))
            self.table_offsets[(schema, table_name)] = {}
            table = self.meta.tables[(schema, table_name)]
            if len(table.primary_key) != 1:
                LOGGER.warning(
                    "Cannot merge multi-column primary key for table %s.%s, ignoring",
                    schema,
                    table_name,
                )
                continue

            pk_col = table.table_obj.columns[table.primary_key[0]]
            if not issubclass(pk_col.type.python_type, int):  # type: ignore
                LOGGER.warning(
                    "Cannot merge non-integer primary key for table %s.%s, ignoring",
                    schema,
                    table_name,
                )
                continue

            with self.engine.connect() as conn:
                max_pk_val = conn.scalar(sa.select(sa.func.max(pk_col)))

            pk_max_values[source_table] = max_pk_val
            if max_pk_val is not None:
                self.table_offsets[(schema, table_name)][pk_col.name] = max_pk_val + 1

        for source_table in self.plan.queries:
            if source_table in self.passthrough_tables:
                continue

            src_schema, src_table_name = parse_table_name(source_table)
            src_table = self.source_meta.tables[(src_schema, src_table_name)]
            for fk in src_table.foreign_keys:
                if len(fk.columns) != 1:
                    continue
                if (
                    fk.dst_columns
                    != self.source_meta.tables[
                        (fk.dst_schema, fk.dst_table)
                    ].primary_key
                ):
                    continue
                offset = pk_max_values.get(f"{fk.dst_schema}.{fk.dst_table}")
                if offset is not None:
                    self.table_offsets[self._remap_table(src_schema, src_table_name)][
                        fk.columns[0]
                    ] = (offset + 1)

    def create(self) -> None:
        """Create any missing tables in destination from the source schema"""
        source_meta = self.source_meta
        metadata_obj = sa.MetaData()

        table_obj_map = {}
        tables_created = set()
        for remapped_table, table in self.table_remap.items():
            remap_schema, remap_table = parse_table_name(remapped_table)

            if (remap_schema, remap_table) in self.meta.tables:
                table_obj_map[table] = self.meta.tables[
                    (remap_schema, remap_table)
                ].table_obj
                continue

            table_obj = source_meta.tables[parse_table_name(table)].table_obj
            table_obj_map[table] = sa.Table(
                remap_table,
                metadata_obj,
                *(
                    sa.Column(
                        col.name,
                        col.type,
                        nullable=col.nullable,
                        primary_key=col.primary_key,
                    )
                    for col in table_obj.columns
                ),
                schema=remap_schema,
            )
            tables_created.add(table_obj_map[table])

        def _remap_cols(cols: Iterable[sa.Column]) -> List[sa.Column]:
            return [
                table_obj_map[f"{col.table.schema}.{col.table.name}"].columns[col.name]
                for col in cols
            ]

        # Copy table constraints including foreign key constraints.
        for table, remapped_table_obj in table_obj_map.items():
            if remapped_table_obj not in tables_created:
                continue

            table_obj = source_meta.tables[parse_table_name(table)].table_obj
            for constraint in table_obj.constraints:
                if isinstance(constraint, sa.UniqueConstraint):
                    remapped_table_obj.append_constraint(
                        sa.UniqueConstraint(*_remap_cols(constraint.columns))
                    )
                if isinstance(constraint, sa.CheckConstraint):
                    remapped_table_obj.append_constraint(
                        sa.CheckConstraint(constraint.sqltext)
                    )
                if isinstance(constraint, sa.ForeignKeyConstraint):
                    fk_cols = _remap_cols(elem.column for elem in constraint.elements)
                    remapped_table_obj.append_constraint(
                        sa.ForeignKeyConstraint(
                            _remap_cols(constraint.columns),
                            fk_cols,
                            name=constraint.name,
                            use_alter=True,
                        )
                    )

            for index_idx, index in enumerate(table_obj.indexes):
                sa.Index(
                    f"idx_subsetter_{remapped_table_obj.name}_{index_idx}",
                    *(
                        (
                            remapped_table_obj.columns[col.name]
                            if isinstance(col, sa.Column)
                            else col
                        )
                        for col in index.columns
                    ),
                    unique=index.unique,
                    dialect_options=index.dialect_options,
                    **index.dialect_kwargs,
                )

        if tables_created:
            LOGGER.info("Creating %d tables in destination", len(tables_created))
            metadata_obj.create_all(bind=self.engine)
            for remapped_table_obj in tables_created:
                self.meta.track_new_table(remapped_table_obj)

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
        (src_schema, src_table_name) = (schema, table_name)
        schema, table_name = self._remap_table(src_schema, src_table_name)
        table = self.meta.tables[(schema, table_name)]

        columns_out = filter_view.columns_out if filter_view else columns
        missing_columns = {
            col for col in columns_out if col not in table.table_obj.columns
        }
        if missing_columns:
            raise ValueError(
                f"Destination table {schema}.{table_name} is missing expected columns {missing_columns}"
            )

        # Automatically omit any included computed columns
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

        offsets = None
        if table_offsets := self.table_offsets.get((schema, table_name)):
            offsets = [
                (index, table_offsets[col])
                for index, col in enumerate(columns_out)
                if col in table_offsets
            ]

        multiplied_indexes = [
            ind
            for ind, col_name in enumerate(columns_out)
            if col_name in (column_multipliers or set())
        ]

        autoinc_index = -1
        autoinc_max_written = None
        if (
            _autoincrement_needs_updating(self.engine)
            and table.table_obj.autoincrement_column is not None
        ):
            try:
                autoinc_index = columns_out.index(
                    table.table_obj.autoincrement_column.name
                )
            except ValueError:
                pass

        buffer: List[tuple] = []

        conflict_strategy = self.conflict_strategy
        if self.merge and f"{src_schema}.{src_table_name}" in self.passthrough_tables:
            conflict_strategy = "skip"

        def _flush_buffer():
            with self.engine.connect() as conn:
                conn.execute(
                    ConflictInsert(conflict_strategy, table.table_obj, columns_out),
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
                if offsets is not None:
                    for index, offset in offsets:
                        if out_row[index] is not None:
                            out_row[index] += offset
                if autoinc_index != -1:
                    if autoinc_max_written is None:
                        autoinc_max_written = out_row[autoinc_index]
                    else:
                        autoinc_max_written = max(
                            autoinc_max_written, out_row[autoinc_index]
                        )
                buffer.append(tuple(out_row))
                if len(buffer) > DESTINATION_BUFFER_SIZE:
                    _flush_buffer()
        if buffer:
            _flush_buffer()

        _autoincrement_update(self.engine, table.table_obj, autoinc_max_written)


class Sampler:
    def __init__(self, source: DatabaseConfig, config: SamplerConfig) -> None:
        self.config = config
        self.source_engine = source.database_engine(env_prefix="SUBSET_SOURCE_")

    def sample(
        self,
        plan: SubsetPlan,
        *,
        truncate: bool = False,
        create: bool = False,
    ) -> None:
        meta, _ = DatabaseMetadata.from_engine(self.source_engine, list(plan.queries))
        if self.config.infer_foreign_keys != "none":
            meta.infer_missing_foreign_keys(
                infer_all=self.config.infer_foreign_keys == "all"
            )
        self._validate_filters(meta)

        table_column_multipliers = self._get_multiplied_columns(meta, plan)

        output = SamplerOutput.from_config(self.config.output, plan, meta)
        if create:
            output.create()
        insert_order = output.insert_order()
        if truncate:
            output.truncate()
        output.prepare()

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
        self, meta: DatabaseMetadata, plan: SubsetPlan
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
        ignore_tables = set(plan.passthrough) | set(
            self.config.multiplicity.ignore_tables
        )

        for table_name in plan.queries:
            if table_name in ignore_tables:
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

            for table_name in plan.queries:
                table = meta.tables[parse_table_name(table_name)]

                cols = result.get(table_name, set())
                start_len = len(cols)
                for fk in table.foreign_keys:
                    dst_mapped = result.get(f"{fk.dst_schema}.{fk.dst_table}", set())
                    cols.update(set(fk.columns) & dst_mapped)

                if cols and table_name in ignore_tables:
                    raise ValueError(
                        "Passthrough foreign key points to multiplied column"
                    )

                changes = changes or len(cols) > start_len

            if not changes:
                break

        for table_name in plan.queries:
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
