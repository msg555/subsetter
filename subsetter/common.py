import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import sqlalchemy as sa
from pydantic import BaseModel

DatabaseDialect = Literal[
    "mysql",
    "postgres",
    "sqlite",
]

LOGGER = logging.getLogger(__name__)
DEFAULT_DIALECT: Literal["mysql"] = "mysql"

SQLKnownOperator = Literal["<", ">", "=", "<>", "!=", "like", "not like"]
SQLLiteralType = Union[str, int, float]

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
        database = database or os.getenv(f"{env_prefix}DATABASE", None)
        username = username or os.getenv(f"{env_prefix}USERNAME", None)
        password = password or os.getenv(f"{env_prefix}PASSWORD", None)

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
    elif dialect == "sqlite":
        return sa.engine.URL.create(drivername="sqlite", database=database)
    else:
        raise ValueError(f"Unsupported SQL dialect {dialect!r}")

    return sa.engine.URL.create(
        drivername=drivername,
        host=host,
        port=port,
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
    class SSLConfig(BaseModel):
        class SSLClientConfig(BaseModel):
            cert: Optional[str] = None
            key: Optional[str] = None

        class SSLServerConfig(BaseModel):
            rootcert: Optional[str] = None
            verify: Literal["none", "cert", "identity"] = "none"

        require: Optional[bool] = False
        client: SSLClientConfig = SSLClientConfig()
        server: SSLServerConfig = SSLServerConfig()

        def get_connect_args(
            self,
            dialect: DatabaseDialect,
            *,
            env_prefix: Optional[str] = None,
        ) -> Dict:
            as_dict = self.model_dump(exclude_unset=True)
            require = as_dict.get(
                "require", bool(os.getenv(f"{env_prefix}SSL_REQUIRE", ""))
            )
            if not require:
                return {}

            server_verify = as_dict.get("server", {}).get(
                "verify", os.getenv(f"{env_prefix}SSL_SERVER_VERIFY", "none")
            )
            client_cert = self.client.cert or os.getenv(f"{env_prefix}SSL_CLIENT_CERT")
            client_key = self.client.key or os.getenv(f"{env_prefix}SSL_CLIENT_KEY")
            server_rootcert = self.server.rootcert or os.getenv(
                f"{env_prefix}SSL_SERVER_ROOTCERT"
            )

            result: Dict
            if dialect == "postgres":
                if server_verify == "identity":
                    sslmode = "verify-full"
                elif server_verify == "cert":
                    sslmode = "verify-ca"
                else:
                    sslmode = "require"

                result = {
                    "sslcert": client_cert,
                    "sslkey": client_key,
                    "sslmode": sslmode,
                    "sslrootcert": server_rootcert,
                }
                return {key: value for key, value in result.items() if value}

            if dialect == "mysql":
                result = {
                    "ca": server_rootcert,
                    "key": client_key,
                    "cert": client_cert,
                }
                result = {key: value for key, value in result.items() if value}
                if server_verify == "identity":
                    result["verify_identity"] = True
                elif server_verify == "cert":
                    result["verify_cert"] = True
                else:
                    result["verify_cert"] = False
                return {"ssl": result}

            raise ValueError("Dialect does not support ssl")

    dialect: Optional[DatabaseDialect] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    session_sqls: List[str] = []
    sqlite_databases: Optional[Dict[str, str]] = {}
    isolation_level: Optional[IsolationLevel] = None
    ssl: SSLConfig = SSLConfig()

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
        if self.isolation_level:
            isolation_level = self.isolation_level
        elif self.dialect == "sqlite":
            isolation_level = "SERIALIZABLE"
        else:
            isolation_level = "READ COMMITTED"

        engine = sa.create_engine(
            self.database_url(env_prefix=env_prefix),
            isolation_level=isolation_level,
            pool_pre_ping=True,
            connect_args=self.ssl.get_connect_args(
                self.dialect or DEFAULT_DIALECT,
                env_prefix=env_prefix,
            ),
        )

        @sa.event.listens_for(engine, "connect")
        def _set_session_sqls(dbapi_connection, _):
            cursor = dbapi_connection.cursor()
            try:
                if self.dialect == "sqlite":
                    for db_alias, db_file in self.sqlite_databases.items():
                        escaped_db_file = db_file.replace("'", "''")
                        cursor.execute(
                            f"ATTACH DATABASE '{escaped_db_file}' as {db_alias}"
                        )

                for session_sql in self.session_sqls:
                    cursor.execute(session_sql)
            finally:
                cursor.close()

        return engine
