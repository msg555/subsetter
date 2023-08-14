import logging
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import sqlalchemy as sa
from pydantic import BaseModel, Field

from subsetter.common import (
    DatabaseConfig,
    database_url,
    mysql_column_list,
    mysql_identifier,
    mysql_table_name,
    parse_table_name,
)
from subsetter.metadata import DatabaseMetadata, ForeignKey, TableMetadata
from subsetter.solver import dfs, order_graph, reverse_graph, subgraph

LOGGER = logging.getLogger(__name__)


class PlannerConfig(BaseModel):
    class TargetConfig(BaseModel):
        percent: Optional[float] = None
        amount: Optional[int] = None
        like: Dict[str, List[str]] = {}
        in_: Dict[str, List[Union[int, str]]] = Field({}, alias="in")

    class ColumnConstraint(BaseModel):
        column: str
        operator: Literal["<", ">", "=", "<>", "!=", "in", "not in"]
        expression: Union[int, float, str, List[Union[int, float, str]]]

    class IgnoreFKConfig(BaseModel):
        src_table: str
        dst_table: str

    class ExtraFKConfig(BaseModel):
        src_table: str
        src_columns: List[str]
        dst_table: str
        dst_columns: List[str]

    source: DatabaseConfig = DatabaseConfig()

    global_constraints: List[ColumnConstraint] = []
    table_constraints: Dict[str, List[ColumnConstraint]] = {}

    select: List[str]
    targets: Dict[str, TargetConfig]
    ignore: List[str] = []
    passthrough: List[str] = []
    ignore_fks: List[IgnoreFKConfig] = []
    extra_fks: List[ExtraFKConfig] = []


class SubsetPlan(BaseModel):
    class SubsetQuery(BaseModel):
        query: str
        params: Dict[str, Union[int, float, str, List[Union[int, float, str]]]] = {}
        materialize: bool = False

    queries: Dict[str, SubsetQuery]


class Planner:
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.engine = sa.create_engine(
            database_url(
                env_prefix="SUBSET_SOURCE_",
                host=self.config.source.host,
                port=self.config.source.port,
                username=self.config.source.username,
                password=self.config.source.password,
            )
        )
        self.meta: DatabaseMetadata
        self.ignore_tables = {parse_table_name(table) for table in config.ignore}
        self.passthrough_tables = {
            parse_table_name(table) for table in config.passthrough
        }

    def plan(self) -> SubsetPlan:
        LOGGER.info("Scanning schema")
        meta, extra_tables = DatabaseMetadata.from_engine(
            self.engine,
            self.config.select,
            close_forward=True,
        )
        self.meta = meta
        LOGGER.info("done!")

        for schema, table_name in meta.tables:
            if (schema, table_name) not in extra_tables:
                LOGGER.info("Selected table %s.%s", schema, table_name)
        for extra_table in extra_tables:
            LOGGER.info(
                "Selected additional table %s.%s referenced by foreign keys",
                extra_table[0],
                extra_table[1],
            )

        self.meta.infer_missing_foreign_keys()
        self._remove_ignore_fks()
        self._add_extra_fks()

        with open("graph.dot", "w", encoding="utf-8") as fgraph:
            self.meta.output_graphviz(fgraph)

        self._check_ignore_tables()
        self._check_passthrough_tables()

        order = self._solve_order()
        self.meta.compute_reverse_keys()

        queries = {}
        for schema, table_name in self.passthrough_tables:
            queries[f"{schema}.{table_name}"] = SubsetPlan.SubsetQuery(
                query=f"SELECT * FROM {mysql_table_name(schema, table_name)}",
            )

        processed = set()
        remaining = {parse_table_name(table) for table in order}
        for table in order:
            schema, table_name = parse_table_name(table)
            processed.add((schema, table_name))
            remaining.remove((schema, table_name))
            queries[table] = self._plan_table(
                self.meta.tables[(schema, table_name)],
                processed,
                remaining,
                target=self.config.targets.get(table),
            )

        return SubsetPlan(queries=queries)

    def _solve_order(self) -> List[str]:
        """
        Attempts to compute an ordering of non-passthrough, non-ignored tables
        satisfies all constraints.
        """
        self.meta.normalize_foreign_keys()

        source = ""
        graph = self.meta.as_graph(
            ignore_tables=self.passthrough_tables | self.ignore_tables
        )
        graph[source] = list(self.config.targets)

        visited: Dict[str, int] = {}
        dfs(reverse_graph(graph, union=True), source, visited=visited)
        for u in graph:
            if u not in visited:
                LOGGER.warning(
                    "warning: no relationship found to %s, ignoring table", u
                )
        graph = subgraph(graph, set(visited))

        return order_graph(graph, source)

    def _add_extra_fks(self) -> None:
        """Add in additional foreign keys requested."""
        for extra_fk in self.config.extra_fks:
            src_schema, src_table_name = parse_table_name(extra_fk.src_table)
            dst_schema, dst_table_name = parse_table_name(extra_fk.dst_table)
            table = self.meta.tables.get((src_schema, src_table_name))
            if table is None:
                LOGGER.warning(
                    "Found no source table %s.%s referenced in add_extra_fks",
                    src_schema,
                    src_table_name,
                )
                continue
            if (dst_schema, dst_table_name) not in self.meta.tables:
                LOGGER.warning(
                    "Found no destination table %s.%s referenced in add_extra_fks",
                    dst_schema,
                    dst_table_name,
                )
                continue

            table.foreign_keys.append(
                ForeignKey(
                    columns=tuple(extra_fk.src_columns),
                    dst_schema=dst_schema,
                    dst_table=dst_table_name,
                    dst_columns=tuple(extra_fk.dst_columns),
                ),
            )

    def _remove_ignore_fks(self) -> None:
        """Remove requested foreign keys"""
        for ignore_fk in self.config.ignore_fks:
            src_schema, src_table_name = parse_table_name(ignore_fk.src_table)
            dst_schema, dst_table_name = parse_table_name(ignore_fk.dst_table)
            table = self.meta.tables.get((src_schema, src_table_name))
            if table is None:
                LOGGER.warning(
                    "Found no table %s.%s referenced in ignore_fks",
                    src_schema,
                    src_table_name,
                )
                continue

            table.foreign_keys = [
                fk
                for fk in table.foreign_keys
                if (fk.dst_schema, fk.dst_table) != (dst_schema, dst_table_name)
            ]

    def _check_ignore_tables(self) -> None:
        """
        Make sure no processed table has an FK into an ignored table.
        """
        for table in self.meta.tables.values():
            if (table.schema, table.name) in self.ignore_tables:
                continue
            for fk in table.foreign_keys:
                if (fk.dst_schema, fk.dst_table) in self.ignore_tables:
                    raise ValueError(
                        f"Foreign key from {table.schema!r}.{table.name!r} into"
                        f" ignored table {fk.dst_schema!r}.{fk.dst_table!r}"
                    )

    def _check_passthrough_tables(self) -> None:
        """
        Make sure no passthrough table has an FK to a non-passthrough table.
        """
        not_found_tables = set()
        for schema, table_name in self.passthrough_tables:
            table = self.meta.tables.get((schema, table_name))
            if table is None:
                not_found_tables.add((schema, table_name))
                LOGGER.warning(
                    "Could not find passthrough table %s.%s", schema, table_name
                )
                continue

            # Ensure that passthrough tables have no foreign keys to tables outside of this set.
            for foreign_key in table.foreign_keys:
                if (
                    foreign_key.dst_schema,
                    foreign_key.dst_table,
                ) not in self.passthrough_tables:
                    raise ValueError(
                        f"Passthrough table {schema!r}.{table_name!r} has foreign key to non passthrough "
                        f"table {foreign_key.dst_schema!r}.{foreign_key.dst_table!r}"
                    )

        self.passthrough_tables -= not_found_tables

    def _plan_table(
        self,
        table: TableMetadata,
        processed: Set[Tuple[str, str]],
        remaining: Set[Tuple[str, str]],
        target: Optional[PlannerConfig.TargetConfig] = None,
    ) -> SubsetPlan.SubsetQuery:
        materialize = False
        or_constraints = []
        final_clause = ""
        for fk in table.foreign_keys + table.rev_foreign_keys:
            dst_key = (fk.dst_schema, fk.dst_table)
            if dst_key not in processed:
                if dst_key in remaining:
                    materialize = True
                continue

            or_constraints.append(
                f"{mysql_column_list(fk.columns)} IN (SELECT {mysql_column_list(fk.dst_columns)} "
                f"FROM {mysql_table_name(fk.dst_schema, fk.dst_table + '<SAMPLED>')})"
            )

        params: Dict[str, Union[int, float, str, List[Union[int, float, str]]]] = {}
        and_constraints = []

        conf_constraints = (
            self.config.global_constraints
            + self.config.table_constraints.get(f"{table.schema}.{table.name}", [])
        )
        for conf_constraint in conf_constraints:
            if conf_constraint.column in table.columns:
                param_name = f"param_{len(params)}"
                and_constraints.append(
                    f"{mysql_identifier(conf_constraint.column)} {conf_constraint.operator} :{param_name}"
                )
                params[param_name] = conf_constraint.expression

        if target is not None:
            if target.percent is not None:
                or_constraints.append(f"rand() < {target.percent / 100.0}")
            elif target.amount is not None:
                final_clause = f" ORDER BY rand() LIMIT {target.amount}"
            else:
                for column, patterns in target.like.items():
                    for pattern in patterns:
                        param_name = f"param_{len(params)}"
                        or_constraints.append(
                            f"{mysql_identifier(column)} LIKE :{param_name}"
                        )
                        params[param_name] = pattern
                for column, in_list in target.in_.items():
                    param_name = f"param_{len(params)}"
                    or_constraints.append(
                        f"{mysql_identifier(column)} IN :{param_name}"
                    )
                    params[param_name] = list(in_list)

        query = f"SELECT * FROM {mysql_table_name(table.schema, table.name)}"
        if or_constraints and and_constraints:
            query = f"{query} WHERE ({' OR '.join(or_constraints)}) AND {' AND '.join(and_constraints)}"
        elif or_constraints:
            query = f"{query} WHERE {' OR '.join(or_constraints)}"
        elif and_constraints:
            query = f"{query} WHERE {' AND '.join(and_constraints)}"

        return SubsetPlan.SubsetQuery(
            query=query + final_clause,
            params=params,
            materialize=materialize,
        )
