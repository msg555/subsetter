import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from subsetter.common import (
    DEFAULT_DIALECT,
    DatabaseConfig,
    SQLKnownOperator,
    SQLLiteralType,
    SQLStatementSelect,
    SQLStatementUnion,
    SQLTableIdentifier,
    SQLTableQuery,
    SQLWhereClause,
    SQLWhereClauseAnd,
    SQLWhereClauseOperator,
    SQLWhereClauseOr,
    SQLWhereClauseRandom,
    SQLWhereClauseSQL,
    parse_table_name,
)
from subsetter.metadata import DatabaseMetadata, ForeignKey, TableMetadata
from subsetter.solver import dfs, order_graph, reverse_graph, subgraph
from subsetter.sql_dialects import SQLDialectEncoder

LOGGER = logging.getLogger(__name__)


class PlannerConfig(BaseModel):
    class TargetConfig(BaseModel):
        all_: bool = Field(False, alias="all")
        percent: Optional[float] = None
        amount: Optional[int] = None
        like: Dict[str, List[str]] = {}
        in_: Dict[str, List[SQLLiteralType]] = Field({}, alias="in")
        sql: Optional[str] = None

    class IgnoreFKConfig(BaseModel):
        src_table: str
        dst_table: str

    class ExtraFKConfig(BaseModel):
        src_table: str
        src_columns: List[str]
        dst_table: str
        dst_columns: List[str]

    class ColumnConstraint(BaseModel):
        column: str
        operator: SQLKnownOperator
        values: Union[SQLLiteralType, List[SQLLiteralType]]

    source: DatabaseConfig = DatabaseConfig()
    table_constraints: Dict[str, List[ColumnConstraint]] = {}
    select: List[str]
    targets: Dict[str, TargetConfig]
    ignore: List[str] = []
    passthrough: List[str] = []
    ignore_fks: List[IgnoreFKConfig] = []
    extra_fks: List[ExtraFKConfig] = []
    infer_foreign_keys: bool = False
    normalize_foreign_keys: bool = False
    output_sql: bool = False


class SubsetPlan(BaseModel):
    queries: Dict[str, SQLTableQuery]


class Planner:
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.engine = self.config.source.database_engine(env_prefix="SUBSET_SOURCE_")
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

        if self.config.infer_foreign_keys:
            self.meta.infer_missing_foreign_keys()
        if self.config.normalize_foreign_keys:
            self.meta.normalize_foreign_keys()
        self._remove_ignore_fks()
        self._add_extra_fks()
        self._check_ignore_tables()
        self._check_passthrough_tables()

        order = self._solve_order()
        self.meta.compute_reverse_keys()

        queries: Dict[str, SQLTableQuery] = {}
        for schema, table_name in self.passthrough_tables:
            queries[f"{schema}.{table_name}"] = SQLTableQuery(
                statement=SQLStatementSelect(
                    type_="select",
                    from_=SQLTableIdentifier(
                        table_schema=schema,
                        table_name=table_name,
                    ),
                )
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

        if self.config.output_sql:
            for table, query in queries.items():
                if not query.statement:
                    continue
                sql_params: Dict[str, Any] = {}
                queries[table] = SQLTableQuery(
                    sql=query.statement.build(
                        SQLDialectEncoder.from_dialect(
                            self.config.source.dialect or DEFAULT_DIALECT
                        ),
                        sql_params,
                    ),
                    sql_params=sql_params,
                    materialize=query.materialize,
                )

        return SubsetPlan(queries=queries)

    def _solve_order(self) -> List[str]:
        """
        Attempts to compute an ordering of non-passthrough, non-ignored tables
        satisfies all constraints.
        """
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
    ) -> SQLTableQuery:
        materialize = False
        fk_constraints: List[SQLWhereClause] = []
        for fk in table.foreign_keys + table.rev_foreign_keys:
            dst_key = (fk.dst_schema, fk.dst_table)
            if dst_key not in processed:
                if dst_key in remaining:
                    dst_target = self.config.targets.get(
                        f"{fk.dst_schema}.{fk.dst_table}"
                    )
                    if not dst_target or not dst_target.all_:
                        materialize = True
                continue

            if not target or not target.all_:
                fk_constraints.append(
                    SQLWhereClauseOperator(
                        type_="operator",
                        operator="in",
                        columns=list(fk.columns),
                        values=SQLStatementSelect(
                            type_="select",
                            columns=list(fk.dst_columns),
                            from_=SQLTableIdentifier(
                                table_schema=fk.dst_schema,
                                table_name=fk.dst_table,
                                sampled=True,
                            ),
                        ),
                    )
                )

        conf_constraints = self.config.table_constraints.get(
            f"{table.schema}.{table.name}", []
        )
        conf_constraints_sql: List[SQLWhereClause] = []
        for conf_constraint in conf_constraints:
            if conf_constraint.column in table.columns:
                conf_constraints_sql.append(
                    SQLWhereClauseOperator(
                        type_="operator",
                        operator=conf_constraint.operator,
                        columns=[conf_constraint.column],
                        values=conf_constraint.values,
                    )
                )

        # Calculate initial foreign-key / config constraint statement
        statements: List[SQLStatementSelect] = [
            SQLStatementSelect(
                type_="select",
                from_=SQLTableIdentifier(
                    table_schema=table.schema,
                    table_name=table.name,
                ),
                where=SQLWhereClauseAnd(
                    type_="and",
                    conditions=[
                        *conf_constraints_sql,
                        SQLWhereClauseOr(
                            type_="or",
                            conditions=fk_constraints,
                        ),
                    ],
                ),
            )
        ]

        # If targetted also calculate target constraint statement
        if target:
            target_constraints: List[SQLWhereClause] = []
            if target.percent is not None:
                target_constraints.append(
                    SQLWhereClauseRandom(
                        type_="random", threshold=target.percent / 100.0
                    )
                )

            if target.sql is not None:
                target_constraints.append(
                    SQLWhereClauseSQL(
                        type_="sql",
                        sql=target.sql,
                    )
                )

            for column, patterns in target.like.items():
                target_constraints.append(
                    SQLWhereClauseOr(
                        type_="or",
                        conditions=[
                            SQLWhereClauseOperator(
                                type_="operator",
                                operator="like",
                                columns=[column],
                                values=pattern,
                            )
                            for pattern in patterns
                        ],
                    )
                )

            for column, in_list in target.in_.items():
                target_constraints.append(
                    SQLWhereClauseOperator(
                        type_="operator",
                        operator="in",
                        columns=[column],
                        values=in_list,
                    )
                )

            if target.all_:
                target_constraints.clear()

            statements.append(
                SQLStatementSelect(
                    type_="select",
                    from_=SQLTableIdentifier(
                        table_schema=table.schema,
                        table_name=table.name,
                    ),
                    where=SQLWhereClauseAnd(type_="and", conditions=target_constraints),
                    limit=target.amount,
                )
            )

        return SQLTableQuery(
            statement=SQLStatementUnion(
                type_="union",
                statements=statements,
            ).simplify(),
            materialize=materialize,
        )
