import logging
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

import sqlalchemy as sa
from pydantic import BaseModel, Field

from subsetter.common import (
    DatabaseConfig,
    mysql_identifier,
    mysql_table_name,
    parse_table_name,
)
from subsetter.filters import FilterConfig, FilterViewChain
from subsetter.metadata import DatabaseMetadata

try:
    from tqdm import tqdm  # type: ignore
except ImportError:

    def tqdm(x, **_):
        return x


LOGGER = logging.getLogger(__name__)

_NOT_SET = object()


def _multiply_column(
    value: Optional[int], multiplier: int, iteration: int
) -> Optional[int]:
    if value is None:
        return None
    return value * multiplier + iteration


class MultiplicityConfig(BaseModel):
    multiplier: int = 1
    infer_foreign_keys: bool = False
    passthrough: List[str] = []
    extra_columns: Dict[str, List[str]] = {}
    ignore_primary_key_columns: Dict[str, List[str]] = {}


class ReplayConfig(BaseModel):
    output: DatabaseConfig = DatabaseConfig()
    filters: Dict[str, List[FilterConfig]] = {}  # type: ignore
    multiplicity: MultiplicityConfig = MultiplicityConfig()
    select: List[str]
    table_primary_key: Dict[str, List[str]] = {}


ReplayConfig.update_forward_refs()


class MaxwellStreamRecord(BaseModel):
    database: str
    table: str
    type_: Literal["update", "insert", "delete"] = Field(..., alias="type")
    data: Dict[str, Any]
    old: Dict[str, Any] = {}


class InternalStreamRecord(BaseModel):
    schema_: str = Field(..., alias="schema")
    table: str
    primary_key: Dict[str, Any]
    values: Tuple[Any, ...]
    updated: Optional[Tuple[int, ...]] = None
    deleted: bool = False


class Replayer:
    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self.engine = self.config.output.database_engine(
            env_prefix="REPLAY_DESTINATION_"
        )

    def _transform_stream(
        self, meta: DatabaseMetadata, stream: Iterable[MaxwellStreamRecord]
    ) -> Iterable[InternalStreamRecord]:
        all_tables = [f"{schema}.{table_name}" for schema, table_name in meta.tables]
        table_column_multipliers = self._get_multiplied_columns(meta, all_tables)
        filter_views = {
            table: FilterViewChain.construct_filter(
                list(meta.tables[parse_table_name(table)].columns),
                self.config.filters.get(table, []),
            )
            for table in all_tables
        }
        column_indexes = {
            table: {
                col_name: index
                for index, col_name in enumerate(
                    meta.tables[parse_table_name(table)].columns
                )
            }
            for table in all_tables
        }
        primary_key_columns = {
            table: self.config.table_primary_key.get(
                table, meta.tables[parse_table_name(table)].primary_key
            )
            for table in all_tables
        }
        primary_key_indexes = {
            table: tuple(
                column_indexes[table][col_name]
                for col_name in primary_key_columns[table]
            )
            for table in all_tables
        }
        table_multiplied_indexes = {
            table: [
                ind
                for ind, col_name in enumerate(
                    meta.tables[parse_table_name(table)].columns
                )
                if col_name in column_multipliers
            ]
            for table, column_multipliers in table_column_multipliers.items()
        }

        for record in tqdm(stream):
            table = f"{record.database}.{record.table}"
            table_meta = meta.tables.get(parse_table_name(table))
            if table_meta is None:
                continue

            row = []
            prior_row = []
            for index, col_name in enumerate(table_meta.columns):
                try:
                    row.append(record.data[col_name])
                except KeyError:
                    LOGGER.warning(
                        "Missing column value for column %r, using null", col_name
                    )
                    row.append(None)
                prior_row.append(record.old.get(col_name, row[-1]))
            updated = tuple(
                index
                for index, col_name in enumerate(table_meta.columns)
                if col_name in record.old
            )

            multiplier = 1
            if multiplied_indexes := table_multiplied_indexes.get(table, []):
                multiplier = self.config.multiplicity.multiplier

            pk_columns = primary_key_columns[table]
            pk_indexes = primary_key_indexes[table]
            filter_view = filter_views.get(table)
            for iteration in range(multiplier):
                out_row = filter_view.filter_view(row) if filter_view else list(row)
                out_prior_row = (
                    filter_view.filter_view(prior_row)
                    if filter_view
                    else list(prior_row)
                )
                for index in multiplied_indexes:
                    out_row[index] = _multiply_column(
                        out_row[index], multiplier, iteration
                    )
                    out_prior_row[index] = _multiply_column(
                        out_prior_row[index], multiplier, iteration
                    )

                yield InternalStreamRecord(
                    schema=record.database,
                    table=record.table,
                    primary_key={
                        col_name: out_prior_row[index]
                        for index, col_name in zip(pk_indexes, pk_columns)
                    },
                    values=tuple(out_row),
                    updated=updated,
                    deleted=record.type_ == "deleted",
                )

    def replay(self, stream: Iterable[MaxwellStreamRecord]) -> None:
        meta, extra_tables = DatabaseMetadata.from_engine(
            self.engine,
            self.config.select,
            close_forward=True,
        )

        for schema, table_name in meta.tables:
            if (schema, table_name) not in extra_tables:
                LOGGER.info("Selected table %s.%s", schema, table_name)
        for extra_table in extra_tables:
            LOGGER.info(
                "Selected additional table %s.%s referenced by foreign keys",
                extra_table[0],
                extra_table[1],
            )

        if self.config.multiplicity.infer_foreign_keys:
            meta.infer_missing_foreign_keys()
        self._validate_filters(meta)

        with self.engine.connect() as conn:
            for record in self._transform_stream(meta, stream):
                where_clause = " AND ".join(
                    f"{mysql_identifier(col_name)}=:pk_{index}"
                    for index, col_name in enumerate(record.primary_key)
                )
                bind = {
                    f"pk_{index}": value
                    for index, value in enumerate(record.primary_key.values())
                }

                if record.deleted:
                    result = conn.execute(
                        sa.text(
                            f"DELETE FROM {mysql_table_name(record.schema_, record.table)} WHERE {where_clause}"
                        ),
                        bind,
                    )
                    conn.commit()
                    if not result.rowcount:
                        LOGGER.warning(
                            "Deleted from %s.%s matched no existing rows",
                            record.schema,
                            record.table,
                        )
                    continue

                columns = list(meta.tables[(record.schema_, record.table)].columns)
                updated = record.updated or tuple(range(len(columns)))

                update_clause = ", ".join(
                    f"{mysql_identifier(columns[index])}=:row_{index}"
                    for index in updated
                )
                bind.update((f"row_{index}", record.values[index]) for index in updated)

                result = conn.execute(
                    sa.text(
                        f"UPDATE {mysql_table_name(record.schema_, record.table)} SET {update_clause} WHERE {where_clause}"
                    ),
                    bind,
                )
                conn.commit()
                if result.rowcount:
                    continue

                column_clause = ", ".join(
                    mysql_identifier(col_name) for col_name in columns
                )
                values_clause = ", ".join(
                    f":row_{index}" for index in range(len(columns))
                )
                bind.update(
                    (f"row_{index}", value) for index, value in enumerate(record.values)
                )
                result = conn.execute(
                    sa.text(
                        f"INSERT INTO {mysql_table_name(record.schema_, record.table)} ({column_clause}) VALUES ({values_clause})"
                    ),
                    bind,
                )
                conn.commit()
                if not result.rowcount:
                    LOGGER.warning("Failed to update or insert row")
                    continue

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


def record_file_stream(file_stream) -> Iterable[MaxwellStreamRecord]:
    for line in file_stream:
        try:
            yield MaxwellStreamRecord.parse_raw(line)
        except ValueError:
            pass


def main():
    # pylint: disable=import-outside-toplevel
    import sys

    import yaml

    with open("replay_config.yaml", "r", encoding="utf-8") as fconfig:
        config = ReplayConfig.parse_obj(yaml.safe_load(fconfig))

    replayer = Replayer(config)
    replayer.replay(record_file_stream(sys.stdin))


if __name__ == "__main__":
    main()
