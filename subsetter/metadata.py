import collections
import dataclasses
import logging
from fnmatch import fnmatch
from typing import Dict, List, Optional, Set, TextIO, Tuple

import sqlalchemy as sa

from subsetter.common import parse_table_name
from subsetter.plan_model import SQLTableIdentifier
from subsetter.solver import toposort

LOGGER = logging.getLogger(__name__)

# We use lists as queues in a couple places. This is well defined by upsets pylint.
# pylint: disable=modified-iterating-list


@dataclasses.dataclass(frozen=True, order=True)
class ForeignKey:
    columns: Tuple[str, ...]
    dst_schema: str
    dst_table: str
    dst_columns: Tuple[str, ...]

    @classmethod
    def from_schema(cls, fk: sa.ForeignKeyConstraint) -> "ForeignKey":
        assert fk.referred_table.schema is not None
        return cls(
            columns=tuple(col.name for col in fk.columns),
            dst_schema=fk.referred_table.schema,
            dst_table=fk.referred_table.name,
            dst_columns=tuple(elem.column.name for elem in fk.elements),
        )


class TableMetadata:
    def __init__(
        self,
        table_obj: sa.Table,
    ) -> None:
        assert table_obj.schema is not None
        self.table_obj = table_obj
        self.schema = table_obj.schema
        self.name = table_obj.name
        self.primary_key = tuple(
            column.name for column in table_obj.primary_key.columns
        )
        self.foreign_keys = [
            ForeignKey.from_schema(fk) for fk in table_obj.foreign_key_constraints
        ]
        self.rev_foreign_keys: List[ForeignKey] = []

    def __str__(self) -> str:
        return f"{self.schema}.{self.name}"


class DatabaseMetadata:
    def __init__(
        self,
        metadata_obj: sa.MetaData,
        tables: Dict[Tuple[str, str], TableMetadata],
        *,
        supports_temp_reopen: bool = True,
    ) -> None:
        self.metadata_obj = metadata_obj
        self.tables = tables
        self.supports_temp_reopen = supports_temp_reopen
        self.temp_tables: Dict[Tuple[str, str, int], sa.Table] = {}

    @classmethod
    def from_engine(
        cls,
        engine,
        select: List[str],
        *,
        close_forward=False,
        close_backward=False,
    ) -> Tuple["DatabaseMetadata", List[Tuple[str, str]]]:

        inspector = sa.inspect(engine)
        metadata_obj = sa.MetaData()
        for schema in inspector.get_schema_names():
            metadata_obj.reflect(bind=engine, schema=schema)

        table_set: Set[Tuple[str, str]] = set()
        table_queue: List[Tuple[str, str]] = []
        for table_id, table_obj in metadata_obj.tables.items():
            assert table_obj.schema is not None
            if any(fnmatch(table_id, select_pattern) for select_pattern in select):
                table_set.add((table_obj.schema, table_obj.name))
                table_queue.append((table_obj.schema, table_obj.name))

        table_deps: Dict[Tuple[str, str], List[Tuple[str, str]]] = (
            collections.defaultdict(list)
        )
        for table_obj in metadata_obj.tables.values():
            assert table_obj.schema is not None
            for fk in table_obj.foreign_key_constraints:
                ref_table_obj = fk.referred_table
                assert ref_table_obj.schema is not None
                if close_forward:
                    table_deps[(table_obj.schema, table_obj.name)].append(
                        (ref_table_obj.schema, ref_table_obj.name)
                    )
                if close_backward:
                    table_deps[(ref_table_obj.schema, ref_table_obj.name)].append(
                        (table_obj.schema, table_obj.name)
                    )

        num_selected_tables = len(table_queue)
        for schema, table in table_queue:
            table_obj = metadata_obj.tables[f"{schema}.{table}"]
            for ref_schema, ref_table in table_deps[(schema, table)]:
                if (ref_schema, ref_table) not in table_set:
                    table_set.add((ref_schema, ref_table))
                    table_queue.append((ref_schema, ref_table))

        return (
            cls(
                metadata_obj,
                {
                    (schema, table): TableMetadata(
                        metadata_obj.tables[f"{schema}.{table}"]
                    )
                    for schema, table in table_queue
                },
                supports_temp_reopen=engine.dialect.name != "mysql",
            ),
            table_queue[num_selected_tables:],
        )

    def infer_missing_foreign_keys(self) -> None:
        pk_map: Dict[Tuple[str, Tuple[str, ...]], Optional[TableMetadata]] = {}
        for table in self.tables.values():
            if not table.primary_key:
                continue
            if table.primary_key in pk_map:
                LOGGER.info("Duplicate primary key columsn found %r", table.primary_key)
                pk_map[(table.schema, table.primary_key)] = None
            else:
                pk_map[(table.schema, table.primary_key)] = table

        # Only infer single column primary keys
        for table in self.tables.values():
            fks = set(table.foreign_keys)
            for col in table.table_obj.columns:
                dst_table = pk_map.get((table.schema, (col.name,)))
                if dst_table is not None and dst_table is not table:
                    fk = ForeignKey(
                        columns=(col.name,),
                        dst_schema=dst_table.schema,
                        dst_table=dst_table.name,
                        dst_columns=(col.name,),
                    )
                    if fk not in fks:
                        LOGGER.info(
                            "Inferring foreign key %s->%s on %r",
                            table,
                            dst_table,
                            fk.columns,
                        )
                        table.foreign_keys.append(fk)

    def normalize_foreign_keys(self) -> None:
        """
        If table A has a foreign key to table B and they both share a foreign
        key on the same column in table C, remove the foreign key from table A
        assuming it is redundant.
        """
        fk_sets = {
            table_key: {
                (fk.dst_schema, fk.dst_table, fk.dst_columns)
                for fk in table.foreign_keys
            }
            for table_key, table in self.tables.items()
        }
        for table in self.tables.values():
            child_fk_sets = set()
            for fk in table.foreign_keys:
                child_fk_sets |= fk_sets[(fk.dst_schema, fk.dst_table)]
            fk_out = []
            for fk in table.foreign_keys:
                if (fk.dst_schema, fk.dst_table, fk.dst_columns) not in child_fk_sets:
                    fk_out.append(fk)
                else:
                    LOGGER.info(
                        "Normalizing foreign key, removed %s->%s.%s on %r",
                        table,
                        fk.dst_schema,
                        fk.dst_table,
                        fk.columns,
                    )
            table.foreign_keys = fk_out

    def toposort(self) -> List[TableMetadata]:
        return [  # type: ignore
            self.tables[parse_table_name(u)] for u in toposort(self.as_graph())
        ]

    def compute_reverse_keys(self) -> None:
        for table in self.tables.values():
            for fk in table.foreign_keys:
                self.tables[(fk.dst_schema, fk.dst_table)].rev_foreign_keys.append(
                    ForeignKey(
                        columns=fk.dst_columns,
                        dst_schema=table.schema,
                        dst_table=table.name,
                        dst_columns=fk.columns,
                    )
                )

    def as_graph(
        self, *, ignore_tables: Optional[Set[Tuple[str, str]]] = None
    ) -> Dict[str, Set[str]]:
        if ignore_tables is None:
            ignore_tables = set()
        return {
            f"{table.schema}.{table.name}": {
                f"{fk.dst_schema}.{fk.dst_table}"
                for fk in table.foreign_keys
                if (fk.dst_schema, fk.dst_table) not in ignore_tables
                and (fk.dst_schema, fk.dst_table) in self.tables
            }
            for table in self.tables.values()
            if (table.schema, table.name) not in ignore_tables
        }

    def sql_build_context(self):
        reference_count = {}

        def _context(ident: SQLTableIdentifier) -> sa.Table:
            if ident.sampled:
                if self.supports_temp_reopen:
                    index = 0
                else:
                    index = reference_count.get(
                        (ident.table_schema, ident.table_name), 0
                    )
                    reference_count[(ident.table_schema, ident.table_name)] = index + 1
                return self.temp_tables[(ident.table_schema, ident.table_name, index)]
            return self.tables[(ident.table_schema, ident.table_name)].table_obj

        return _context

    def output_graphviz(self, fout: TextIO) -> None:
        def _dot_label(lbl: TableMetadata) -> str:
            return f'"{str(lbl)}"'

        fout.write("digraph {\n")
        for table in self.tables.values():
            fout.write("  ")
            fout.write(_dot_label(table))
            fout.write(" -> {")

            deps = {
                self.tables[(fk.dst_schema, fk.dst_table)] for fk in table.foreign_keys
            }
            fout.write(", ".join(_dot_label(dep) for dep in deps))
            fout.write("}\n")
        fout.write("}\n")
