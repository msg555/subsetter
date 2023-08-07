import collections
import abc
import os
import json
import itertools
import sys
from typing import Dict, Optional, Tuple, List, Set, Any

import sqlalchemy as sa
import yaml
from pydantic import BaseModel
from subsetter.config import SubsetConfig, TargetConfig, DirectoryOutputConfig, MysqlOutputConfig
from subsetter.metadata import DatabaseMetadata, ForeignKey, TableMetadata
from subsetter.solver import order_graph, subgraph, reverse_graph, dfs


def _mysql_identifier(identifier: str) -> str:
    assert '`' not in identifier
    return f"`{identifier}`"


def _mysql_table_name(schema: str, table: str) -> str:
    return f"{_mysql_identifier(schema)}.{_mysql_identifier(table)}"


def _mysql_column_list(columns: List[str]) -> str:
    return ", ".join(_mysql_identifier(column) for column in columns)


def _parse_table_name(full_table_name: str) -> Tuple[str, str]:
    parts = full_table_name.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Table name {full_table_name!r} contains no schema")
    return tuple(parts)


def _database_url(
    *,
    env_prefix: Optional[str] = None,
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
        drivername="mysql+pymysql",
        host=host,
        port=int(port),
        username=username,
        password=password,
        query={"charset": "utf8mb4"},
    )


class SubsetOutput(abc.ABC):
    @abc.abstractmethod
    def output_result_set(self, schema: str, table_name: str, columns: List[str], result_set) -> None:
        pass

    def truncate(self, schema: str, table_name: str) -> None:
        pass

    @staticmethod
    def from_destination_config(config: Any) -> "SubsetOutput":
        cls = {
            DirectoryOutputConfig: DirectoryOutput,
            MysqlOutputConfig: MysqlOutput,
        }.get(type(config), None)
        if cls is None:
            raise ValueError("Unexpected config type")
        return cls(config)


class DirectoryOutput(SubsetOutput):
    def __init__(self, config: DirectoryOutputConfig) -> None:
        self.directory = config.directory

    def output_result_set(self, schema: str, table_name: str, columns: List[str], result_set) -> None:
        output_path = os.path.join(self.directory, f"{schema}.{table_name}.json")
        with open(output_path, "w", encoding="utf-8") as fdump:
            for row in result_set:
                json.dump(dict(zip(columns, row)), fdump, default=str)
                fdump.write("\n")


class MysqlOutput(SubsetOutput):
    def __init__(self, config: MysqlOutputConfig) -> None:
        self.engine = sa.create_engine(
            _database_url(
                env_prefix="SUBSET_DESTINATION_",
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
            )
        )

    def truncate(self, schema: str, table_name: str) -> None:
        print(f"Truncating {schema!r}.{table_name!r}")
        with self.engine.connect() as conn:
            conn.execute(sa.text(f"DELETE FROM {_mysql_table_name(schema, table_name)} WHERE 1"))
            conn.commit()

    def output_result_set(self, schema: str, table_name: str, columns: List[str], result_set) -> None:
        insert_query = (
            f"INSERT INTO {_mysql_table_name(schema, table_name)} "
            f"({_mysql_column_list(columns)}) VALUES "
        )
        with self.engine.connect() as conn:
            flush_count = 1024
            buffer = []

            def _flush_buffer():
                query = f"{insert_query}{','.join(f':{ind}' for ind in range(len(buffer)))}"
                conn.execute(
                    sa.text(query),
                    {str(ind): row for ind, row in enumerate(buffer)},
                )
                buffer.clear()

            for row in result_set:
                buffer.append(tuple(row))
                if len(buffer) > flush_count:
                    _flush_buffer()
            if buffer:
                _flush_buffer()

            conn.commit()
    

class Subsetter:
    def __init__(self, config: SubsetConfig) -> None:
        self.config = config
        self.engine = sa.create_engine(
            _database_url(
                env_prefix="SUBSET_SOURCE_",
                host=self.config.source.host,
                port=self.config.source.port,
                username=self.config.source.username,
                password=self.config.source.password,
            )
        )
        self.meta: DatabaseMetadata
        self.ignore_tables = {_parse_table_name(table) for table in config.ignore}
        self.passthrough_tables = {_parse_table_name(table) for table in config.passthrough}
        self.output = SubsetOutput.from_destination_config(config.destination)

    def subset(self) -> None:
        print("Scanning source schemas...", end="", flush=True)
        self.meta = DatabaseMetadata.from_engine(self.engine, self.config.schemas)

        # TODO: Fix doing this before below toposort. Problem with test db
        self._remove_ignore_fks()
        self._add_extra_fks()

        true_table_order = self.meta.toposort()
        print(" done!")

        self._check_ignore_tables()
        self._check_passthrough_tables()

        # TODO: remove
        with open("graph.dot", "w", encoding="utf-8") as fgraph:
            self.meta.output_graphviz(fgraph)

        # TODO: Allow externally specifiying order
        order = self._solve_order()

        self.meta.compute_reverse_keys()

        for table in reversed(true_table_order):
            if (
                (table.schema, table.name) in self.passthrough_tables or
                f"{table.schema}.{table.name}" in order
            ):
                self.output.truncate(table.schema, table.name)

        output_queries = {}
        for schema, table_name in self.passthrough_tables:
            output_queries[(schema, table_name)] = f"SELECT * FROM {_mysql_table_name(schema, table_name)}"

        processed = set()
        remaining = {_parse_table_name(table) for table in order}
        with self.engine.connect() as conn:
            for table in order:
                schema, table_name = _parse_table_name(table)
                processed.add((schema, table_name))
                remaining.remove((schema, table_name))
                output_queries[(schema, table_name)] = self._subset_table(
                    conn,
                    self.meta.tables[(schema, table_name)],
                    processed,
                    remaining,
                    target=self.config.targets.get(table),
                )

            for table in true_table_order:
                query = output_queries.get((table.schema, table.name))
                if query is not None:
                    self._output_table(conn, table, query)

    def _solve_order(self) -> List[Tuple[str, str]]:
        """
        Attempts to compute an ordering of non-passthrough, non-ignored tables
        satisfies all constraints.
        """
        # TODO: Make optional
        self.meta.normalize_foreign_keys()

        source = ""
        graph = self.meta.as_graph(ignore_tables=self.passthrough_tables | self.ignore_tables)
        graph[source] = list(self.config.targets)

        visited = {}
        dfs(reverse_graph(graph, union=True), source, visited=visited)
        for u in graph:
            if u not in visited:
                print(f"warning: no relationship found to {u}, ignoring table")
        graph = subgraph(graph, set(visited))

        # TODO: remove
        with open("graph.json", "w", encoding="utf-8") as fgraph:
            json.dump(graph, fgraph, indent=2)

        return order_graph(graph, source)

    def _add_extra_fks(self) -> None:
        """ Add in additional foreign keys requested. """
        for extra_fk in self.config.extra_fks:
            src_schema, src_table_name = _parse_table_name(extra_fk.src_table)
            dst_schema, dst_table_name = _parse_table_name(extra_fk.dst_table)
            self.meta.add_foreign_key(
                src_schema,
                src_table_name,
                ForeignKey(
                    columns=tuple(extra_fk.src_columns),
                    dst_schema=dst_schema,
                    dst_table=dst_table_name,
                    dst_columns=tuple(extra_fk.dst_columns),
                ),
            )

    def _remove_ignore_fks(self) -> None:
        """ Remove requested foreign keys """
        for ignore_fk in self.config.ignore_fks:
            src_schema, src_table_name = _parse_table_name(ignore_fk.src_table)
            dst_schema, dst_table_name = _parse_table_name(ignore_fk.dst_table)
            table = self.meta.tables[(src_schema, src_table_name)]
            table.foreign_keys = [
                fk for fk in table.foreign_keys
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
                        f" ignored table {fk.dst_schema!r}.{fk.dst_name!r}"
                    )

    def _check_passthrough_tables(self) -> None:
        """
        Make sure no passthrough table has an FK to a non-passthrough table.
        """
        for schema, table in self.passthrough_tables:
            # Ensure that passthrough tables have no foreign keys to tables outside of this set.
            for foreign_key in self.meta.tables[(schema, table)].foreign_keys:
                if (foreign_key.dst_schema, foreign_key.dst_table) not in self.passthrough_tables:
                    raise ValueError(
                        f"Passthrough table {schema!r}.{table!r} has foreign key to non passthrough "
                        f"table {foreign_key.dst_schema!r}.{foreign_key.dst_table!r}"
                    )

    def _output_table(
        self,
        conn,
        table: TableMetadata,
        query: str,
    ) -> None:
        print("MAKING QUERY", query)
        result = conn.execute(sa.text(query))

        rows = 0
        def _count_rows():
            nonlocal rows
            for row in result:
                rows += 1
                yield row

        self.output.output_result_set(table.schema, table.name, table.columns, _count_rows())
        print(f"Subset {rows} rows for {table}")

    def _subset_table(
        self,
        conn,
        table: TableMetadata,
        processed: Set[Tuple[str, str]],
        remaining: Set[Tuple[str, str]],
        target: Optional[TargetConfig] = None,
    ) -> str:
        need_temp = False
        constraints = []
        for fk in table.foreign_keys + table.rev_foreign_keys:
            dst_key = (fk.dst_schema, fk.dst_table)
            if dst_key not in processed:
                if dst_key in remaining:
                    need_temp = True
                continue

            constraints.append(
                f"{_mysql_column_list(fk.columns)} IN (SELECT {_mysql_column_list(fk.dst_columns)} "
                f"FROM {_mysql_table_name(fk.dst_schema, fk.dst_table + '_tmp')})"
            )

        if target is not None:
            constraints.append(f"rand() < {target.percent / 100.0}")
        if not constraints:
            constraints.append("1")

        query = f"SELECT * FROM {_mysql_table_name(table.schema, table.name)} WHERE {' OR '.join(constraints)}"
        if need_temp:
            query = f"CREATE TEMPORARY TABLE {_mysql_table_name(table.schema, table.name + '_tmp')} {query}"

            print("Creating temp lookup table for", table)
            try:
                result = conn.execute(sa.text(query))
            except Exception as exc:
                # Some client/server combinations report a read-only error even though the temporary
                # table creation actually succeeded. We'll just swallow the error here and if there
                # was a real issue it'll get flagged again when we query against it.
                if "--read-only" not in str(exc):
                    raise

            return f"SELECT * FROM {_mysql_table_name(table.schema, table.name + '_tmp')}"
        else:
            return query


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as fconfig:
        config = SubsetConfig.parse_obj(yaml.safe_load(fconfig))

    subsetter = Subsetter(config)
    subsetter.subset()


if __name__ == "__main__":
    main()
