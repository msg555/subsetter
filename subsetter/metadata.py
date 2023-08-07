import collections
import copy
import dataclasses
from typing import List, Optional, Set, Dict, Tuple
import io

import sqlalchemy as sa
from subsetter.solver import reverse_graph


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
    ) -> None:
        self.schema = schema
        self.name = name
        self.columns = columns
        self.primary_key = primary_key
        self.foreign_keys: List[ForeignKey] = []
        self.rev_foreign_keys: List[ForeignKey] = []

    def __str__(self) -> str:
        return f"{self.schema}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.schema}.{self.name}"


class DatabaseMetadata:
    def __init__(self):
        self.tables: Dict[Tuple[str, str], TableMetadata] = {}

    @classmethod
    def from_engine(cls, engine, schemas: List[str]) -> "DatabaseMetadata":
        meta = cls()
        inspector = sa.inspect(engine)

        for schema in schemas:
            for table_name in inspector.get_table_names(schema=schema):
                meta.tables[(schema, table_name)] = TableMetadata(
                    schema=schema,
                    name=table_name,
                    columns=tuple(column["name"] for column in inspector.get_columns(table_name, schema=schema)),
                    primary_key=tuple(inspector.get_pk_constraint(table_name, schema=schema).get("constrained_columns", [])),
                )

        for table in meta.tables.values():
            for fk in inspector.get_foreign_keys(table.name, schema=table.schema):
                meta.add_foreign_key(
                    table.schema,
                    table.name,
                    ForeignKey(
                        columns=tuple(fk["constrained_columns"]),
                        dst_schema=fk["referred_schema"],
                        dst_table=fk["referred_table"],
                        dst_columns=tuple(fk["referred_columns"]),
                    ),
                )

        return meta

    def infer_missing_foreign_keys(self) -> None:
        pk_map = {}
        for table in self.tables.values():
            if not table.primary_key:
                continue
            if table.primary_key in pk_map:
                print(f"Duplicate primary key columsn found {table.primary_key!r}")
                pk_map[(table.schema, table.primary_key)] = None
            else:
                pk_map[(table.schema, table.primary_key)] = table

        # Only infer single column primary keys
        for table in self.tables.values():
            fks = set(table.foreign_keys)
            for col in table.columns:
                dst_table = pk_map.get((table.schema, (col, )))
                if dst_table is not None and dst_table is not table:
                    fk = ForeignKey(
                        columns=(col, ),
                        dst_schema=dst_table.schema,
                        dst_table=dst_table.name,
                        dst_columns=(col, ),
                    )
                    if fk not in fks:
                        print(f"Inferring foreign key {table}->{dst_table} on {fk.columns!r}")
                        table.foreign_keys.append(fk)

    def add_foreign_key(
        self, schema: str, table_name: str, foreign_key: ForeignKey
    ) -> None:
        self.tables[(schema, table_name)].foreign_keys.append(foreign_key)

    def normalize_foreign_keys(self) -> None:
        # First find topological sort of tables
        order = self.toposort()

        fk_sets = {
            table_key: {(fk.dst_schema, fk.dst_table, fk.dst_columns) for fk in table.foreign_keys}
            for table_key, table in self.tables.items()
        }
        for table in self.tables.values():
            child_fk_sets = set()
            for fk in table.foreign_keys:
                child_fk_sets |= fk_sets[(fk.dst_schema, fk.dst_table)]
            table.foreign_keys = [
                fk for fk in table.foreign_keys
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
        return [self.tables[tuple(u.split(".", 1))] for u in order]

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

    def as_graph(self, *, ignore_tables: Optional[Set[str]] = None) -> Dict[str, List[str]]:
        if ignore_tables is None:
            ignore_tables = set()
        return {
            f"{table.schema}.{table.name}": [
                f"{fk.dst_schema}.{fk.dst_table}" for fk in table.foreign_keys
                if (fk.dst_schema, fk.dst_table) not in ignore_tables
            ]
            for table in self.tables.values()
            if (table.schema, table.name) not in ignore_tables
        }


    def output_graphviz(self, fout: io.StringIO) -> None:
        def _dot_label(lbl: TableMetadata) -> str:
            return f'"{str(lbl)}"'

        fout.write("digraph {\n")
        for table in self.tables.values():
            fout.write("  ")
            fout.write(_dot_label(table))
            fout.write(" -> {")

            deps = {self.tables[(fk.dst_schema, fk.dst_table)] for fk in table.foreign_keys}
            fout.write(", ".join(_dot_label(dep) for dep in deps))
            fout.write("}\n")
        fout.write("}\n")
