import os

import pytest
import yaml

from subsetter.common import DatabaseConfig
from subsetter.planner import Planner, PlannerConfig, SubsetPlan
from subsetter.sampler import DatabaseOutputConfig, Sampler, SamplerConfig

from .dataset_manager import apply_dataset, get_rows

DATASETS_BASE_PATH = os.path.join(os.path.dirname(__file__), "datasets")


def db_config_mysql(request):
    mysql = request.getfixturevalue("mysql_proc")
    return DatabaseConfig(
        dialect="mysql",
        host=mysql.host,
        port=mysql.port,
        username=mysql.user,
        password="",
    )


def db_config_postgres(request):
    postgresql = request.getfixturevalue("postgresql_proc")
    return DatabaseConfig(
        dialect="postgres",
        host=postgresql.host,
        port=postgresql.port,
        username=postgresql.user,
        password="",
    )


DATABASE_CONFIGURATIONS = [
    pytest.param(
        db_config_mysql,
        marks=[
            pytest.mark.usefixtures("mysql_proc"),
            pytest.mark.mysql_live,
        ],
        id="mysql",
    ),
    pytest.param(
        db_config_postgres,
        marks=[
            pytest.mark.usefixtures("postgresql_proc"),
            pytest.mark.postgres_live,
        ],
        id="postgres",
    ),
]


@pytest.fixture(name="db_config")
def fixture_db_config(request):
    return request.param(request)


@pytest.mark.parametrize("db_config", DATABASE_CONFIGURATIONS, indirect=True)
def test_basic_subset(db_config):
    apply_dataset(db_config, "user_orders")
    config = PlannerConfig(
        source=db_config,
        targets={
            "test.users": PlannerConfig.TargetConfig(  # type: ignore
                **{"in": {"sample": [1]}}
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

    sample_config = SamplerConfig(
        source=db_config,
        output=DatabaseOutputConfig(
            mode="database",
            remap=[
                {
                    "search": r"^(\w+)\.",
                    "replace": r"\1_out.",
                },
            ],
            **db_config.model_dump(),
        ),
    )

    sampler = Sampler(sample_config)
    sampler.sample(plan)

    sample = {
        "users": get_rows(db_config, "test_out", "users"),
        "orders": get_rows(db_config, "test_out", "orders"),
        "order_status": get_rows(db_config, "test_out", "order_status"),
    }

    expected_sample_file = os.path.join(
        os.path.dirname(__file__), "basic_sample_expected.yml"
    )
    with open(expected_sample_file, "r", encoding="utf-8") as fexpected:
        expected_sample = yaml.safe_load(fexpected)

    assert sample == expected_sample
