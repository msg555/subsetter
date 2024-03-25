import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from subsetter.sql_dialects import DatabaseDialect, SQLDialectEncoder

LOGGER = logging.getLogger(__name__)
DEFAULT_DIALECT: Literal["mysql"] = "mysql"

# pylint: disable=unused-argument


def parse_table_name(full_table_name: str) -> Tuple[str, str]:
    parts = full_table_name.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Table name {full_table_name!r} contains no schema")
    return (parts[0], parts[1])


def database_url(
    *,
    dialect: Optional[DatabaseDialect] = None,
    env_prefix: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> sa.engine.URL:
    if env_prefix is not None:
        dialect = dialect or os.getenv(f"{env_prefix}DIALECT", None)  # type: ignore
        host = host or os.getenv(f"{env_prefix}HOST", "localhost")
        port = port or int(os.getenv(f"{env_prefix}PORT", "0"))
        database = database or os.getenv(f"{env_prefix}DATABASE", "")
        username = username or os.environ[f"{env_prefix}USERNAME"]
        password = os.environ[f"{env_prefix}PASSWORD"] if password is None else password

    if dialect is None:
        dialect = DEFAULT_DIALECT
        LOGGER.warning("No database dialect selected, defaulting to '%s'", dialect)

    extra_kwargs: Dict[str, Any] = {}
    if dialect == "mysql":
        drivername = "mysql+pymysql"
        if not port:
            port = 3306
        extra_kwargs["query"] = {"charset": "utf8mb4"}
    elif dialect == "postgres":
        drivername = "postgresql+psycopg2"
        if not port:
            port = 5432
        if database:
            extra_kwargs["database"] = database
    else:
        raise ValueError(f"Unsupported SQL dialect {dialect!r}")

    return sa.engine.URL.create(
        drivername=drivername,
        host=host,
        port=port or 3306,
        username=username,
        password=password,
        **extra_kwargs,
    )


IsolationLevel = Literal[
    "READ COMMITTED",
    "READ UNCOMMITTED",
    "REPEATABLE READ",
    "SERIALIZABLE",
]


class DatabaseConfig(BaseModel):
    dialect: Optional[DatabaseDialect] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    session_sqls: List[str] = []
    isolation_level: IsolationLevel = "READ COMMITTED"

    def database_url(
        self,
        env_prefix: Optional[str] = None,
    ) -> sa.engine.URL:
        return database_url(
            env_prefix=env_prefix,
            dialect=self.dialect,
            host=self.host,
            port=self.port,
            database=self.database,
            username=self.username,
            password=self.password,
        )

    def database_engine(
        self,
        env_prefix: Optional[str] = None,
    ) -> sa.engine.Engine:
        engine = sa.create_engine(
            self.database_url(env_prefix=env_prefix),
            isolation_level=self.isolation_level,
            pool_pre_ping=True,
        )

        @sa.event.listens_for(engine, "connect")
        def _set_session_sqls(dbapi_connection, _):
            with dbapi_connection.cursor() as cursor:
                for session_sql in self.session_sqls:
                    cursor.execute(session_sql)

        return engine


def _set_param(params_out: Dict[str, Any], param: Any) -> str:
    param_id = f"param{len(params_out)}"
    assert param_id not in params_out
    params_out[param_id] = param
    return f":{param_id}"


class SQLTableIdentifier(BaseModel):
    table_schema: str = Field(..., alias="schema")
    table_name: str = Field(..., alias="table")
    sampled: bool = False

    model_config = ConfigDict(populate_by_name=True)

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        return sql_enc.table_name(
            self.table_schema, self.table_name, sampled=self.sampled
        )


SQLKnownOperator = Literal["<", ">", "=", "<>", "!=", "in", "not in", "like"]
SQLLiteralType = Union[str, int, float]


class SQLWhereClauseFalse(BaseModel):
    type_: Literal["false"] = Field(..., alias="type")

    model_config = ConfigDict(populate_by_name=True)

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        return "0=1"

    def simplify(self) -> "SQLWhereClause":
        return self


class SQLWhereClauseTrue(BaseModel):
    type_: Literal["true"] = Field(..., alias="type")

    model_config = ConfigDict(populate_by_name=True)

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        return "1=1"

    def simplify(self) -> "SQLWhereClause":
        return self


class SQLWhereClauseOperator(BaseModel):
    type_: Literal["operator"] = Field(..., alias="type")
    operator: SQLKnownOperator = "="
    columns: List[str]
    values: Union[SQLLiteralType, List[SQLLiteralType], "SQLStatement"]

    model_config = ConfigDict(populate_by_name=True)

    # TODO: Validate that columns is non empty
    # TODO: Validate that cardinalities match

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        result = [f"{sql_enc.column_list(self.columns)} {self.operator.upper()} "]
        if isinstance(self.values, (SQLStatementSelect, SQLStatementUnion)):
            result.append("(")
            result.append(self.values.build(sql_enc, params_out))
            result.append(")")
        else:
            result.append(_set_param(params_out, self.values))
        return "".join(result)

    def simplify(self) -> "SQLWhereClause":
        return self


class SQLWhereClauseAnd(BaseModel):
    type_: Literal["and"] = Field(..., alias="type")
    conditions: List["SQLWhereClause"]

    model_config = ConfigDict(populate_by_name=True)

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        if not self.conditions:
            return "1=1"
        return " AND ".join(cond.build(sql_enc, params_out) for cond in self.conditions)

    def simplify(self) -> "SQLWhereClause":
        simp_conditions: List["SQLWhereClause"] = [
            simp_condition
            for condition in self.conditions
            if not isinstance(
                simp_condition := condition.simplify(), SQLWhereClauseTrue
            )
        ]
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

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        if not self.conditions:
            return "1=0"
        return " OR ".join(cond.build(sql_enc, params_out) for cond in self.conditions)

    def simplify(self) -> "SQLWhereClause":
        simp_conditions: List["SQLWhereClause"] = [
            simp_condition
            for condition in self.conditions
            if not isinstance(
                simp_condition := condition.simplify(), SQLWhereClauseFalse
            )
        ]
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

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        return f"{sql_enc.random()} < {_set_param(params_out, self.threshold)}"

    def simplify(self) -> "SQLWhereClause":
        return self


class SQLWhereClauseSQL(BaseModel):
    type_: Literal["sql"] = Field(..., alias="type")
    sql: str

    model_config = ConfigDict(populate_by_name=True)

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        return f"({self.sql})"

    def simplify(self) -> "SQLWhereClause":
        return self


SQLWhereClause = Annotated[
    Union[
        SQLWhereClauseOperator,
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

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        result = ["SELECT "]
        if self.columns:
            result.append(sql_enc.column_list(self.columns))
        else:
            result.append("*")

        result.append(" FROM ")
        result.append(self.from_.build(sql_enc, params_out))

        if self.where:
            result.append(" WHERE ")
            result.append(self.where.build(sql_enc, params_out))

        if self.limit is not None:
            result.append(f" ORDER BY {sql_enc.random()} LIMIT {self.limit}")

        return "".join(result)

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

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        return " UNION DISTINCT ".join(
            f"({statement.build(sql_enc, params_out)})" for statement in self.statements
        )

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
    materialize: bool = False

    def build(self, sql_enc: SQLDialectEncoder, params_out: Dict[str, Any]) -> str:
        if self.sql is not None:
            if self.sql_params:
                params_out.update(self.sql_params)
            return self.sql
        if self.statement is not None:
            return self.statement.build(sql_enc, params_out)
        raise ValueError("One of 'sql' or 'select' must be set")
