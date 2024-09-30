import itertools
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from subsetter.common import SQLKnownOperator, SQLLiteralType

# pylint: disable=unused-argument


SQLBuildContext = Callable[["SQLTableIdentifier"], sa.Table]


class SQLTableIdentifier(BaseModel):
    table_schema: str = Field(..., alias="schema")
    table_name: str = Field(..., alias="table")
    sampled: bool = False

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext) -> sa.Table:
        return context(self)


class SQLWhereClauseFalse(BaseModel):
    type_: Literal["false"] = Field(..., alias="type")

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext, table_obj: sa.Table):
        return sa.false()

    def simplify(self) -> "SQLWhereClause":
        return self


class SQLWhereClauseTrue(BaseModel):
    type_: Literal["true"] = Field(..., alias="type")

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext, table_obj: sa.Table):
        return sa.true()

    def simplify(self) -> "SQLWhereClause":
        return self


class SQLWhereClauseOperator(BaseModel):
    type_: Literal["operator"] = Field(..., alias="type")
    operator: SQLKnownOperator = "="
    column: str
    value: SQLLiteralType

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext, table_obj: sa.Table):
        op = self.operator
        column = table_obj.columns[self.column]
        if op == "<":
            return column < self.value
        if op == ">":
            return column > self.value
        if op == "=":
            return column == self.value
        if op in ("<>", "!="):
            return column != self.value
        if op == "like":
            return column.like(self.value)
        if op == "not like":
            return sa.not_(column.like(self.value))
        raise ValueError(f"Unknown operator {op!r}")

    def simplify(self) -> "SQLWhereClause":
        return self


class SQLWhereClauseIn(BaseModel):
    type_: Literal["in"] = Field(..., alias="type")
    columns: List[str]
    values: Union[List[List[SQLLiteralType]], "SQLStatement"]
    negated: bool = False

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext, table_obj: sa.Table):
        columns = sa.tuple_(*(table_obj.columns[col_name] for col_name in self.columns))
        if isinstance(self.values, list):
            clause = columns.in_(self.values)
        else:
            clause = columns.in_(self.values.build(context))
        if self.negated:
            clause = sa.not_(clause)
        return clause

    def simplify(self) -> "SQLWhereClause":
        if isinstance(self.values, list):
            return self
        return SQLWhereClauseIn(
            type_="in",
            columns=list(self.columns),
            values=self.values.simplify(),
            negated=self.negated,
        )


class SQLWhereClauseAnd(BaseModel):
    type_: Literal["and"] = Field(..., alias="type")
    conditions: List["SQLWhereClause"]

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext, table_obj: sa.Table):
        if not self.conditions:
            return sa.true()
        return sa.and_(*(cond.build(context, table_obj) for cond in self.conditions))

    def simplify(self) -> "SQLWhereClause":
        simp_conditions: List["SQLWhereClause"] = [
            simp_condition
            for condition in self.conditions
            if not isinstance(
                simp_condition := condition.simplify(), SQLWhereClauseTrue
            )
        ]
        simp_conditions = list(
            itertools.chain(
                *(
                    cond.conditions if isinstance(cond, SQLWhereClauseAnd) else (cond,)
                    for cond in simp_conditions
                )
            )
        )
        if any(
            isinstance(condition, SQLWhereClauseFalse) for condition in simp_conditions
        ):
            return SQLWhereClauseFalse(type_="false")
        if len(simp_conditions) == 0:
            return SQLWhereClauseTrue(type_="true")
        if len(simp_conditions) == 1:
            return simp_conditions[0]
        return SQLWhereClauseAnd(type_="and", conditions=simp_conditions)


class SQLWhereClauseOr(BaseModel):
    type_: Literal["or"] = Field(..., alias="type")
    conditions: List["SQLWhereClause"]

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext, table_obj: sa.Table):
        if not self.conditions:
            return sa.false()
        return sa.or_(*(cond.build(context, table_obj) for cond in self.conditions))

    def simplify(self) -> "SQLWhereClause":
        simp_conditions: List["SQLWhereClause"] = [
            simp_condition
            for condition in self.conditions
            if not isinstance(
                simp_condition := condition.simplify(), SQLWhereClauseFalse
            )
        ]
        simp_conditions = list(
            itertools.chain(
                *(
                    cond.conditions if isinstance(cond, SQLWhereClauseOr) else (cond,)
                    for cond in simp_conditions
                )
            )
        )
        if any(
            isinstance(condition, SQLWhereClauseTrue) for condition in simp_conditions
        ):
            return SQLWhereClauseTrue(type_="true")
        if len(simp_conditions) == 0:
            return SQLWhereClauseFalse(type_="false")
        if len(simp_conditions) == 1:
            return simp_conditions[0]
        return SQLWhereClauseOr(type_="or", conditions=simp_conditions)


class SQLWhereClauseRandom(BaseModel):
    type_: Literal["random"] = Field(..., alias="type")
    threshold: float

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext, table_obj: sa.Table):
        # pylint: disable=not-callable
        return sa.func.random() < self.threshold

    def simplify(self) -> "SQLWhereClause":
        if self.threshold <= 0:
            return SQLWhereClauseFalse(type_="false")
        if self.threshold >= 1:
            return SQLWhereClauseTrue(type_="true")
        return self


class SQLWhereClauseSQL(BaseModel):
    type_: Literal["sql"] = Field(..., alias="type")
    sql: str
    sql_params: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext, table_obj: sa.Table):
        clause = sa.text(self.sql)
        if self.sql_params is not None:
            clause = clause.bindparams(**self.sql_params)
        return clause

    def simplify(self) -> "SQLWhereClause":
        return self


SQLWhereClause = Annotated[
    Union[
        SQLWhereClauseOperator,
        SQLWhereClauseIn,
        SQLWhereClauseAnd,
        SQLWhereClauseOr,
        SQLWhereClauseRandom,
        SQLWhereClauseSQL,
        SQLWhereClauseTrue,
        SQLWhereClauseFalse,
    ],
    Field(..., discriminator="type_"),
]


class SQLStatementSelect(BaseModel):
    type_: Literal["select"] = Field(..., alias="type")
    columns: Optional[List[str]] = None
    from_: SQLTableIdentifier = Field(..., alias="from")
    where: Optional[SQLWhereClause] = None
    limit: Optional[int] = None

    model_config = ConfigDict(populate_by_name=True)

    def build(self, context: SQLBuildContext):
        table_obj = self.from_.build(context)

        if self.columns:
            # pylint: disable=not-an-iterable
            stmt = sa.select(*(table_obj.columns[column] for column in self.columns))
        else:
            stmt = sa.select(table_obj)

        if self.where:
            stmt = stmt.where(self.where.build(context, table_obj))

        if self.limit is not None:
            # pylint: disable=not-callable
            stmt = stmt.order_by(sa.func.random()).limit(self.limit)

        return stmt

    def simplify(self) -> "SQLStatementSelect":
        if not self.where:
            return self
        simp_where = self.where.simplify()
        kwargs = {
            "type_": self.type_,
            "from_": self.from_,
        }
        if not isinstance(simp_where, SQLWhereClauseTrue):
            kwargs["where"] = simp_where
        if self.columns is not None:
            kwargs["columns"] = self.columns
        if self.limit is not None:
            kwargs["limit"] = self.limit
        return SQLStatementSelect(**kwargs)  # type: ignore


class SQLStatementUnion(BaseModel):
    type_: Literal["union"] = Field(..., alias="type")
    statements: List[SQLStatementSelect]

    model_config = ConfigDict(populate_by_name=True)
    # TODO: Assert statements is not empty

    def build(self, context: SQLBuildContext):
        return sa.union(*(statement.build(context) for statement in self.statements))

    def simplify(self) -> "SQLStatement":
        simp_statements = [
            simp_statement
            for statement in self.statements
            if not isinstance(
                (simp_statement := statement.simplify()).where, SQLWhereClauseFalse
            )
        ]

        if not simp_statements:
            return self.statements[0].simplify()

        statement0 = simp_statements[0]
        if len(simp_statements) == 1:
            return statement0

        for statement in simp_statements:
            if statement.from_ != statement0.from_:
                break
            if statement.columns != statement0.columns:
                break
            if statement.limit is not None:
                break
        else:
            return SQLStatementSelect(
                type_="select",
                from_=statement0.from_,
                columns=statement0.columns,
                where=SQLWhereClauseOr(
                    type_="or",
                    conditions=[
                        (statement.where or SQLWhereClauseTrue(type_="true"))
                        for statement in simp_statements
                    ],
                ),
            ).simplify()

        return SQLStatementUnion(type_="union", statements=simp_statements)


SQLStatement = Annotated[
    Union[
        SQLStatementSelect,
        SQLStatementUnion,
    ],
    Field(..., discriminator="type_"),
]


class SQLTableQuery(BaseModel):
    # TODO: Validate exactly one is set
    sql: Optional[str] = None
    sql_params: Optional[Dict[str, Any]] = None
    statement: Optional[SQLStatement] = None

    def build(self, context: SQLBuildContext):
        if self.sql is not None:
            stmt = sa.text(self.sql)
            if self.sql_params is not None:
                stmt = stmt.bindparams(**self.sql_params)
            return stmt
        if self.statement is not None:
            return self.statement.build(context)
        raise ValueError("One of 'sql' or 'select' must be set")


class SubsetPlan(BaseModel):
    queries: Dict[str, SQLTableQuery]
    passthrough: List[str] = []
