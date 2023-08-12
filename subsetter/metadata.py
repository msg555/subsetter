import collections
import dataclasses
import itertools
import logging
from fnmatch import fnmatch
from typing import Dict, List, Optional, Set, TextIO, Tuple

import sqlalchemy as sa

from subsetter.common import parse_table_name
from subsetter.solver import reverse_graph

LOGGER = logging.getLogger(__name__)

# We use lists as queues in a couple places. This is well defined by upsets pylint.
# pylint: disable=modified-iterating-list


@dataclasses.dataclass(frozen=True)
class ForeignKey:
    columns: Tuple[str, ...]
    dst_schema: str
    dst_table: str
    dst_columns: Tuple[str, ...]


class TableMetadata:
    def __init__(
        self,
        *,
        schema: str,
        name: str,
        columns: Tuple[str, ...],
        primary_key: Tuple[str, ...],
        foreign_keys: List[ForeignKey],
    ) -> None:
        self.schema = schema
        self.name = name
        self.columns = columns
        self.primary_key = primary_key
        self.foreign_keys = list(foreign_keys)
        self.rev_foreign_keys: List[ForeignKey] = []

    def __str__(self) -> str:
        return f"{self.schema}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.schema}.{self.name}"


def _get_fks(engine) -> Dict[Tuple[str, str], List[ForeignKey]]:
    @dataclasses.dataclass(order=True)
    class FKConstraint:
        ordinal_position: int
        table_schema: str
        table_name: str
        column_name: str
        dst_table_schema: str
        dst_table_name: str
        dst_column_name: str

    fk_constraints: Dict[Tuple[str, str], List[FKConstraint]] = collections.defaultdict(
        list
    )
    with engine.connect() as conn:
        result = conn.execute(
            sa.text(
                "SELECT constraint_schema, constraint_name, ordinal_position, "
                "table_schema, table_name, column_name, "
                "referenced_table_schema, referenced_table_name, referenced_column_name "
                "FROM information_schema.key_column_usage WHERE referenced_table_name is not NULL"
            )
        )
        for row in result:
            constraint_schema = row[0]
            constraint_name = row[1]
            fk_constraints[(constraint_schema, constraint_name)].append(
                FKConstraint(*row[2:9])
            )

    all_fks: Dict[Tuple[str, str], List[ForeignKey]] = collections.defaultdict(list)
    for cnst_list in fk_constraints.values():
        cnst_list.sort()

        # Assert that ordinal positions make sense
        assert all(
            cnst.ordinal_position == ind + 1 for ind, cnst in enumerate(cnst_list)
        )

        src_schema = cnst_list[0].table_schema
        src_table = cnst_list[0].table_name
        dst_schema = cnst_list[0].dst_table_schema
        dst_table = cnst_list[0].dst_table_name

        # Assert that all constraints refer to the same pair of tables
        assert all(
            (
                cnst.table_schema,
                cnst.table_name,
                cnst.dst_table_schema,
                cnst.dst_table_name,
            )
            == (src_schema, src_table, dst_schema, dst_table)
            for cnst in cnst_list
        )

        all_fks[(src_schema, src_table)].append(
            ForeignKey(
                columns=tuple(cnst.column_name for cnst in cnst_list),
                dst_schema=dst_schema,
                dst_table=dst_table,
                dst_columns=tuple(cnst.dst_column_name for cnst in cnst_list),
            )
        )

    return all_fks


class DatabaseMetadata:
    def __init__(self) -> None:
        self.tables: Dict[Tuple[str, str], TableMetadata] = {}

    @classmethod
    def from_engine(
        cls,
        engine,
        select: List[str],
        *,
        close_forward=False,
        close_backward=False,
    ) -> Tuple["DatabaseMetadata", List[Tuple[str, str]]]:
        all_fks = _get_fks(engine)
        all_rev_fks = collections.defaultdict(list)
        for (schema, table_name), foreign_keys in all_fks.items():
            for foreign_key in foreign_keys:
                all_rev_fks[(foreign_key.dst_schema, foreign_key.dst_table)].append(
                    ForeignKey(
                        columns=foreign_key.dst_columns,
                        dst_schema=schema,
                        dst_table=table_name,
                        dst_columns=foreign_key.columns,
                    )
                )

        meta = cls()
        inspector = sa.inspect(engine)

        table_queue = []
        table_set = set()
        for schema in inspector.get_schema_names():
            for table_name in inspector.get_table_names(schema=schema):
                if not any(
                    fnmatch(f"{schema}.{table_name}", select_pattern)
                    for select_pattern in select
                ):
                    continue
                table_set.add((schema, table_name))
                table_queue.append((schema, table_name))
        num_selected_tables = len(table_queue)

        for schema, table_name in table_queue:
            table = TableMetadata(
                schema=schema,
                name=table_name,
                columns=tuple(
                    column["name"]
                    for column in inspector.get_columns(table_name, schema=schema)
                ),
                primary_key=tuple(
                    inspector.get_pk_constraint(table_name, schema=schema).get(
                        "constrained_columns", []
                    )
                ),
                foreign_keys=all_fks[(schema, table_name)],
            )

            for foreign_key in itertools.chain(
                all_fks[(table.schema, table.name)] if close_forward else (),
                all_rev_fks[(table.schema, table.name)] if close_backward else (),
            ):
                if (foreign_key.dst_schema, foreign_key.dst_table) not in table_set:
                    table_set.add((foreign_key.dst_schema, foreign_key.dst_table))
                    table_queue.append((foreign_key.dst_schema, foreign_key.dst_table))

            meta.tables[(schema, table_name)] = table

        return meta, table_queue[num_selected_tables:]

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
            for col in table.columns:
                dst_table = pk_map.get((table.schema, (col,)))
                if dst_table is not None and dst_table is not table:
                    fk = ForeignKey(
                        columns=(col,),
                        dst_schema=dst_table.schema,
                        dst_table=dst_table.name,
                        dst_columns=(col,),
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
            table.foreign_keys = [
                fk
                for fk in table.foreign_keys
                if (fk.dst_schema, fk.dst_table, fk.dst_columns) not in child_fk_sets
            ]

    def toposort(self) -> List[TableMetadata]:
        graph = self.as_graph()
        graph_rev = reverse_graph(graph)
        deg = {u: set(edges) for u, edges in graph.items()}
        order = []

        for u, edges in deg.items():
            if not edges:
                order.append(u)

        for u in order:
            for v in graph_rev[u]:
                deg[v].discard(u)
                if not deg[v]:
                    order.append(v)

        if len(order) < len(graph):
            raise ValueError("Cycle detected in schema graph")
        return [self.tables[parse_table_name(u)] for u in order]

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
    ) -> Dict[str, List[str]]:
        if ignore_tables is None:
            ignore_tables = set()
        return {
            f"{table.schema}.{table.name}": [
                f"{fk.dst_schema}.{fk.dst_table}"
                for fk in table.foreign_keys
                if (fk.dst_schema, fk.dst_table) not in ignore_tables
                and (fk.dst_schema, fk.dst_table) in self.tables
            ]
            for table in self.tables.values()
            if (table.schema, table.name) not in ignore_tables
        }

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
