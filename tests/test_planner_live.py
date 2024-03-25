import os

import pytest
import yaml

from subsetter.common import DatabaseConfig
from subsetter.planner import Planner, PlannerConfig, SubsetPlan

from .dataset_manager import apply_dataset

DATASETS_BASE_PATH = os.path.join(os.path.dirname(__file__), "datasets")


@pytest.fixture(name="mysql_db_config")
def fixture_mysql_db_config(mysql_proc) -> DatabaseConfig:
    return DatabaseConfig(
        dialect="mysql",
        host=mysql_proc.host,
        port=mysql_proc.port,
        username=mysql_proc.user,
        password="",
    )


@pytest.fixture(name="postgres_db_config")
def fixture_postgres_db_config(postgresql_proc) -> DatabaseConfig:
    return DatabaseConfig(
        dialect="postgres",
        host=postgresql_proc.host,
        port=postgresql_proc.port,
        username=postgresql_proc.user,
        password="",
    )


@pytest.mark.mysql_live
def test_basic_plan_mysql(mysql_db_config: DatabaseConfig):
    apply_dataset(mysql_db_config, "user_orders")
    config = PlannerConfig(
        source=mysql_db_config,
        targets={
            "test.users": PlannerConfig.TargetConfig(  # type: ignore
                amount=10,
            ),
        },
        select=[
            "test.*",
        ],
    )
    planner = Planner(config)
    plan = planner.plan()

    expected_plan_file = os.path.join(
        os.path.dirname(__file__), "basic_plan_expected.yml"
    )
    with open(expected_plan_file, "r", encoding="utf-8") as fexpected:
        expected_plan = SubsetPlan.model_validate(yaml.safe_load(fexpected))

    assert plan == expected_plan


@pytest.mark.postgres_live
def test_basic_plan_postgres(postgres_db_config: DatabaseConfig):
    apply_dataset(postgres_db_config, "user_orders")
    config = PlannerConfig(
        source=postgres_db_config,
        targets={
            "test.users": PlannerConfig.TargetConfig(  # type: ignore
                amount=10,
            ),
        },
        select=[
            "test.*",
        ],
    )
    planner = Planner(config)
    plan = planner.plan()

    expected_plan_file = os.path.join(
        os.path.dirname(__file__), "basic_plan_expected.yml"
    )
    with open(expected_plan_file, "r", encoding="utf-8") as fexpected:
        expected_plan = SubsetPlan.model_validate(yaml.safe_load(fexpected))

    assert plan == expected_plan
