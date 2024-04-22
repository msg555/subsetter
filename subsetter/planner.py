import logging
from typing import Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from subsetter.common import DatabaseConfig, parse_table_name
from subsetter.metadata import DatabaseMetadata, ForeignKey, TableMetadata
from subsetter.plan_model import (
    SQLKnownOperator,
    SQLLiteralType,
    SQLStatementSelect,
    SQLStatementUnion,
    SQLTableIdentifier,
    SQLTableQuery,
    SQLWhereClause,
    SQLWhereClauseAnd,
    SQLWhereClauseIn,
    SQLWhereClauseOperator,
    SQLWhereClauseOr,
    SQLWhereClauseRandom,
    SQLWhereClauseSQL,
    SubsetPlan,
)
from subsetter.solver import order_graph

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
        value: Union[SQLLiteralType, List[SQLLiteralType]]

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
        for table in order:
            schema, table_name = parse_table_name(table)
            processed.add((schema, table_name))
            queries[table] = self._plan_table(
                self.meta.tables[(schema, table_name)],
                processed,
                target=self.config.targets.get(table),
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
        graph[source] = set(self.config.targets)
        order = order_graph(graph, source)[1:]

        order_st = set(order)
        for table in graph:
            if table and table not in order_st:
                LOGGER.warning(
                    "warning: no relationship found to %s, ignoring table", table
                )

        return order

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
        target: Optional[PlannerConfig.TargetConfig] = None,
    ) -> SQLTableQuery:
        fk_constraints: List[SQLWhereClause] = []

        foreign_keys = sorted(
            fk
            for fk in table.foreign_keys
            if (fk.dst_schema, fk.dst_table) in processed
        )
        rev_foreign_keys = sorted(
            fk
            for fk in table.rev_foreign_keys
            if (fk.dst_schema, fk.dst_table) in processed
        )

        # Make sure the solver gave us something reasonable
        assert not foreign_keys or not rev_foreign_keys
        if target:
            assert not foreign_keys
            if target.all_:
                rev_foreign_keys.clear()

        fk_constraints = [
            SQLWhereClauseIn(
                type_="in",
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
            for fk in foreign_keys or rev_foreign_keys
        ]

        fk_constraint: SQLWhereClause
        if foreign_keys:
            fk_constraint = SQLWhereClauseAnd(
                type_="and",
                conditions=fk_constraints,
            )
        else:
            fk_constraint = SQLWhereClauseOr(
                type_="or",
                conditions=fk_constraints,
            )

        conf_constraints = self.config.table_constraints.get(
            f"{table.schema}.{table.name}", []
        )
        conf_constraints_sql: List[SQLWhereClause] = []
        all_columns = {column.name for column in table.table_obj.columns}
        for conf_constraint in conf_constraints:
            if conf_constraint.column in all_columns:
                conf_constraints_sql.append(
                    SQLWhereClauseOperator(
                        type_="operator",
                        operator=conf_constraint.operator,
                        column=conf_constraint.column,
                        value=conf_constraint.value,
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
                        fk_constraint,
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
                                column=column,
                                value=pattern,
                            )
                            for pattern in patterns
                        ],
                    )
                )

            for column, in_list in target.in_.items():
                target_constraints.append(
                    SQLWhereClauseIn(
                        type_="in",
                        columns=[column],
                        values=[in_list],
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
            ).simplify()
        )
