import os
import sys
from typing import Any, Dict, List, Union

import sqlalchemy as sa
import yaml
from pydantic import BaseModel, ConfigDict, Field

from subsetter.common import DatabaseConfig, parse_table_name
from subsetter.planner import Planner, PlannerConfig, SubsetPlan
from subsetter.sampler import DatabaseOutputConfig, Sampler, SamplerConfig

DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
DATASET_BASE_PATH = os.path.join(DATA_BASE_PATH, "datasets")


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

        def make_table(
            self, metadata: sa.MetaData, schema: str, name: str, schema_suffix: str = ""
        ) -> sa.Table:
            table = sa.Table(
                name,
                metadata,
                schema=schema + schema_suffix,
                *(_col_spec(col) for col in self.columns),
            )
            if self.primary_key:
                table.append_constraint(sa.PrimaryKeyConstraint(*self.primary_key))
            for fk in self.foreign_keys:
                dst_schema, dst_table = parse_table_name(fk.dst_table)
                table.append_constraint(
                    sa.ForeignKeyConstraint(
                        fk.columns,
                        [
                            f"{dst_schema}{schema_suffix}.{dst_table}.{ref_col}"
                            for ref_col in fk.dst_columns
                        ],
                    )
                )
            return table

    tables: Dict[str, TableDescriptor]
    data: Dict[str, List[List[Any]]]


class TestConfig(BaseModel):
    dataset: str
    plan_config: PlannerConfig
    sample_config: SamplerConfig
    expected_plan: SubsetPlan
    expected_sample: Dict[str, List[Dict[str, Any]]]


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
    if col_desc.type_ == "json":
        return sa.Column(col_desc.name, sa.JSON)
    raise ValueError(f"Unknown type {col_desc.type_}")


def apply_dataset(db_config: DatabaseConfig, dataset: TestDataset) -> None:
    engine = sa.create_engine(db_config.database_url())

    schemas = set()
    metadata = sa.MetaData()
    for table, table_config in dataset.tables.items():
        schema, table_name = parse_table_name(table)
        table_config.make_table(metadata, schema, table_name)
        table_config.make_table(metadata, schema, table_name, schema_suffix="_out")
        schemas.add(schema)
        schemas.add(schema + "_out")

    with engine.connect() as conn:
        for schema in schemas:
            try:
                conn.execute(sa.schema.DropSchema(schema, cascade=True, if_exists=True))
            except sa.exc.ProgrammingError:
                conn.execute(sa.schema.DropSchema(schema, if_exists=True))
            conn.execute(sa.schema.CreateSchema(schema))
        conn.commit()

    metadata.create_all(engine)

    with engine.connect() as conn:
        for table, rows in dataset.data.items():
            conn.execute(sa.insert(metadata.tables[table]).values(rows))

        conn.commit()


def get_rows(db_config, schema: str, table: str) -> List[Dict[str, Any]]:
    engine = sa.create_engine(db_config.database_url())
    with engine.connect() as conn:
        metadata_obj = sa.MetaData()
        table_obj = sa.Table(table, metadata_obj, schema=schema, autoload_with=conn)
        return [
            {str(key): val for key, val in row.items()}
            for row in conn.execute(sa.select(table_obj)).mappings()
        ]


def do_dataset_test(db_config: DatabaseConfig, test_name: str) -> None:
    data_path = os.path.join(DATA_BASE_PATH, f"{test_name}.yaml")
    with open(data_path, "r", encoding="utf-8") as fdata:
        test_config = TestConfig.model_validate(yaml.safe_load(fdata))

    dataset_path = os.path.join(DATASET_BASE_PATH, f"{test_config.dataset}.yaml")
    with open(dataset_path, "r", encoding="utf-8") as fdataset:
        dataset = TestDataset.model_validate(yaml.safe_load(fdataset))

    apply_dataset(db_config, dataset)

    test_config.plan_config.source = db_config
    planner = Planner(test_config.plan_config)
    plan = planner.plan()

    if plan != test_config.expected_plan:
        print("Computed plan:")
        yaml.dump(
            plan.dict(exclude_unset=True, by_alias=True),
            stream=sys.stdout,
            default_flow_style=False,
            width=2**20,
            sort_keys=True,
        )
        assert plan == test_config.expected_plan, "Got unexpected plan"

    test_config.sample_config.source = db_config
    test_config.sample_config.output = DatabaseOutputConfig(
        mode="database",
        remap=[
            {
                "search": r"^(\w+)\.",
                "replace": r"\1_out.",
            },
        ],
        **db_config.model_dump(),
    )

    sampler = Sampler(test_config.sample_config)
    sampler.sample(plan)

    sample = {
        table: get_rows(db_config, *parse_table_name(table))
        for table in test_config.expected_sample
    }

    if sample != test_config.expected_sample:
        print("Computed sample:")
        yaml.dump(
            sample,
            stream=sys.stdout,
            default_flow_style=False,
            width=2**20,
            sort_keys=True,
        )
        assert sample == test_config.expected_sample, "Got unexpected sample"
