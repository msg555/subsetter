import os

import pytest

from subsetter.common import DatabaseConfig
from subsetter.planner import Planner, PlannerConfig

from .dataset_manager import apply_dataset

DATASETS_BASE_PATH = os.path.join(os.path.dirname(__file__), "datasets")


@pytest.fixture(name="db_config")
def fixture_db_config(mysql_proc) -> DatabaseConfig:
    return DatabaseConfig(
        host=mysql_proc.host,
        port=mysql_proc.port,
        username=mysql_proc.user,
        password="",
    )


@pytest.mark.mysql_live
def test_basic_plan(db_config: DatabaseConfig):
    apply_dataset(db_config, "user_orders")
    config = PlannerConfig(
        source=db_config,
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
    assert list(plan.queries) == ["test.users", "test.orders", "test.order_status"]
    assert (
        plan.queries["test.users"].query
        == "SELECT * FROM `test`.`users` ORDER BY rand() LIMIT 10"
    )
    assert plan.queries["test.users"].materialize
    assert (
        plan.queries["test.orders"].query
        == "SELECT * FROM `test`.`orders` WHERE `user_id` IN (SELECT `id` FROM `test`.`users<SAMPLED>`)"
    )
    assert (
        plan.queries["test.order_status"].query
        == "SELECT * FROM `test`.`order_status` WHERE `order_id` IN (SELECT `id` FROM `test`.`orders<SAMPLED>`)"
    )
