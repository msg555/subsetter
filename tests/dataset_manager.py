import os
from typing import Any, Dict, List

import sqlalchemy as sa
import yaml
from pydantic import BaseModel

from subsetter.common import (
    DatabaseConfig,
    mysql_identifier,
    mysql_table_name,
    parse_table_name,
)

DATASETS_BASE_PATH = os.path.join(os.path.dirname(__file__), "datasets")


class TestDataset(BaseModel):
    class TableDescriptor(BaseModel):
        class ForeignKeyDescriptor(BaseModel):
            columns: List[str]
            dst_table: str
            dst_columns: List[str]

        columns: List[str]
        primary_key: List[str] = []
        foreign_keys: List[ForeignKeyDescriptor] = []

    databases: List[str]
    tables: Dict[str, TableDescriptor]
    data: Dict[str, List[List[Any]]]


def apply_dataset(db_config: DatabaseConfig, name: str) -> None:
    dataset_path = os.path.join(DATASETS_BASE_PATH, f"{name}.yaml")
    with open(dataset_path, "r", encoding="utf-8") as fdataset:
        config = TestDataset.parse_obj(yaml.safe_load(fdataset))
    engine = sa.create_engine(db_config.database_url())

    def _col_spec(column: str) -> str:
        col_parts = column.split("|", 1)
        col_name = col_parts[0]
        col_type = col_parts[1] if len(col_parts) > 1 else "int"
        mysql_type: str
        if col_type == "str":
            mysql_type = "TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
        elif col_type == "int":
            mysql_type = "int(11)"
        else:
            assert f"Unknown type {mysql_type}"
        return f"{mysql_identifier(col_name)} {mysql_type}"

    with engine.connect() as conn:
        for database in config.databases:
            conn.execute(
                sa.text(f"DROP DATABASE IF EXISTS {mysql_identifier(database)}")
            )
            conn.execute(sa.text(f"CREATE DATABASE {mysql_identifier(database)}"))
        for table, table_config in config.tables.items():
            schema, table_name = parse_table_name(table)
            col_specs = [_col_spec(column) for column in table_config.columns]
            if table_config.primary_key:
                col_specs.append(
                    "PRIMARY KEY ("
                    + ",".join(
                        mysql_identifier(colname)
                        for colname in table_config.primary_key
                    )
                    + ")"
                )
            for foreign_key in table_config.foreign_keys:
                dst_schema, dst_table_name = parse_table_name(foreign_key.dst_table)
                col_specs.append(
                    "FOREIGN KEY ("
                    + ",".join(
                        mysql_identifier(colname) for colname in foreign_key.columns
                    )
                    + f") REFERENCES {mysql_table_name(dst_schema, dst_table_name)}("
                    + ",".join(
                        mysql_identifier(colname) for colname in foreign_key.dst_columns
                    )
                    + ")"
                )

            create_sql = (
                f"CREATE TABLE {mysql_table_name(schema, table_name)} ("
                + ", ".join(col_specs)
                + ")"
            )
            conn.execute(sa.text(create_sql))
        conn.commit()
