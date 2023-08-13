import os
from typing import Iterable, Optional, Tuple

import sqlalchemy as sa
from pydantic import BaseModel


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
        password = password or os.environ[f"{env_prefix}PASSWORD"]
    return sa.engine.URL.create(
        drivername=drivername,
        host=host,
        port=port or 3306,
        username=username,
        password=password,
        query={"charset": "utf8mb4"},
    )


class DatabaseConfig(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
