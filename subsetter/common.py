import os
from typing import Iterable, List, Literal, Optional, Tuple

import sqlalchemy as sa
from pydantic import BaseModel

_NOT_SET = object()


def mysql_identifier(identifier: str) -> str:
    assert "`" not in identifier
    return f"`{identifier}`"


def mysql_table_name(schema: str, table: str) -> str:
    return f"{mysql_identifier(schema)}.{mysql_identifier(table)}"


def mysql_column_list(columns: Iterable[str]) -> str:
    return ", ".join(mysql_identifier(column) for column in columns)


def parse_table_name(full_table_name: str) -> Tuple[str, str]:
    parts = full_table_name.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Table name {full_table_name!r} contains no schema")
    return (parts[0], parts[1])


def database_url(
    *,
    env_prefix: Optional[str] = None,
    drivername="mysql+pymysql",
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> sa.engine.URL:
    if env_prefix is not None:
        host = host or os.getenv(f"{env_prefix}HOST", "localhost")
        port = port or int(os.getenv(f"{env_prefix}PORT", "3306"))
        username = username or os.environ[f"{env_prefix}USERNAME"]
        password = os.environ[f"{env_prefix}PASSWORD"] if password is None else password
    return sa.engine.URL.create(
        drivername=drivername,
        host=host,
        port=port or 3306,
        username=username,
        password=password,
        query={"charset": "utf8mb4"},
    )


IsolationLevel = Literal[
    "AUTOCOMMIT",
    "READ COMMITTED",
    "READ UNCOMMITTED",
    "REPEATABLE READ",
    "SERIALIZABLE",
]


class DatabaseConfig(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    session_sqls: List[str] = []
    isolation_level: IsolationLevel = "AUTOCOMMIT"

    def database_url(
        self,
        env_prefix: Optional[str] = None,
        drivername="mysql+pymysql",
    ) -> sa.engine.URL:
        return database_url(
            env_prefix=env_prefix,
            drivername=drivername,
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )

    def database_engine(
        self,
        env_prefix: Optional[str] = None,
        drivername="mysql+pymysql",
    ) -> sa.engine.Engine:
        engine = sa.create_engine(
            self.database_url(env_prefix=env_prefix, drivername=drivername),
            isolation_level=self.isolation_level,
            pool_pre_ping=True,
        )

        @sa.event.listens_for(engine, "connect")
        def _set_session_sqls(dbapi_connection, _):
            with dbapi_connection.cursor() as cursor:
                for session_sql in self.session_sqls:
                    cursor.execute(session_sql)

        return engine
