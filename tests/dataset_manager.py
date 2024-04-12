import os
from typing import Any, Dict, List, Union

import sqlalchemy as sa
import yaml
from pydantic import BaseModel, ConfigDict, Field

from subsetter.common import (
    DEFAULT_DIALECT,
    DatabaseConfig,
    SQLStatementSelect,
    SQLTableIdentifier,
    parse_table_name,
)
from subsetter.sql_dialects import SQLDialectEncoder

DATASETS_BASE_PATH = os.path.join(os.path.dirname(__file__), "datasets")


class ColumnDescriptor(BaseModel):
    name: str
    type_: str = Field("int", alias="type")

    model_config = ConfigDict(populate_by_name=True)


class TestDataset(BaseModel):
    class TableDescriptor(BaseModel):
        class ForeignKeyDescriptor(BaseModel):
            columns: List[str]
            dst_table: str
            dst_columns: List[str]

        columns: List[Union[str, ColumnDescriptor]]
        primary_key: List[str] = []
        foreign_keys: List[ForeignKeyDescriptor] = []

        def make_table(self, metadata: sa.MetaData, schema: str, name: str) -> sa.Table:
            table = sa.Table(
                name,
                metadata,
                schema=schema,
                *(_col_spec(col) for col in self.columns),
            )
            if self.primary_key:
                table.append_constraint(sa.PrimaryKeyConstraint(*self.primary_key))
            for fk in self.foreign_keys:
                table.append_constraint(
                    sa.ForeignKeyConstraint(
                        fk.columns,
                        [f"{fk.dst_table}.{ref_col}" for ref_col in fk.dst_columns],
                    )
                )
            return table

    tables: Dict[str, TableDescriptor]
    data: Dict[str, List[List[Any]]]


def _col_spec(col: Union[str, ColumnDescriptor]) -> sa.Column:
    if isinstance(col, ColumnDescriptor):
        col_desc = col
    else:
        col_parts = col.split("|", 1)
        col_desc = ColumnDescriptor(
            name=col_parts[0],
            type_=col_parts[1] if len(col_parts) > 1 else "int",
        )
    if col_desc.type_ == "str":
        return sa.Column(col_desc.name, sa.Text)
    if col_desc.type_ == "int":
        return sa.Column(col_desc.name, sa.Integer)
    raise ValueError(f"Unknown type {col_desc.type_}")


def apply_dataset(db_config: DatabaseConfig, name: str) -> None:
    dialect = db_config.dialect or DEFAULT_DIALECT
    sql_enc = SQLDialectEncoder.from_dialect(dialect)

    dataset_path = os.path.join(DATASETS_BASE_PATH, f"{name}.yaml")
    with open(dataset_path, "r", encoding="utf-8") as fdataset:
        config = TestDataset.model_validate(yaml.safe_load(fdataset))
    engine = sa.create_engine(db_config.database_url())

    schemas = set()
    metadata = sa.MetaData()
    for table, table_config in config.tables.items():
        schema, table_name = parse_table_name(table)
        table_config.make_table(metadata, schema, table_name)
        table_config.make_table(metadata, schema + "_out", table_name)
        schemas.add(schema)
        schemas.add(schema + "_out")

    with engine.connect() as conn:
        if db_config.dialect == "mysql":
            for schema in schemas:
                conn.execute(
                    sa.text(f"DROP DATABASE IF EXISTS {sql_enc.identifier(schema)}")
                )
                conn.execute(sa.text(f"CREATE DATABASE {sql_enc.identifier(schema)}"))
        if db_config.dialect == "postgres":
            for schema in schemas:
                conn.execute(
                    sa.text(
                        f"DROP SCHEMA IF EXISTS {sql_enc.identifier(schema)} CASCADE"
                    )
                )
                conn.execute(sa.text(f"CREATE SCHEMA {sql_enc.identifier(schema)}"))

        conn.commit()

    metadata.create_all(engine)

    with engine.connect() as conn:
        for table, rows in config.data.items():
            schema, table_name = parse_table_name(table)
            columns = [_col_spec(col).name for col in config.tables[table].columns]
            for row in rows:
                conn.execute(
                    sa.text(
                        f"INSERT INTO {sql_enc.table_name(schema, table_name)} "
                        f"({sql_enc.column_list(columns)}) VALUES :data"
                    ),
                    {"data": tuple(row)},
                )

        conn.commit()


def get_rows(db_config, schema: str, table: str) -> List[Dict[str, Any]]:
    dialect = db_config.dialect or DEFAULT_DIALECT
    sql_enc = SQLDialectEncoder.from_dialect(dialect)

    engine = sa.create_engine(db_config.database_url())
    with engine.connect() as conn:
        stmt = SQLStatementSelect(
            type_="select",
            from_=SQLTableIdentifier(
                table_schema=schema,
                table_name=table,
            ),
        )
        params: Dict[str, Any] = {}
        sql = stmt.build(sql_enc, params)
        result = conn.execute(sa.text(sql), params)

        return [dict(row) for row in result.mappings()]
