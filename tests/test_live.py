import os

import pytest

from subsetter.common import DatabaseConfig

from .dataset_manager import do_dataset_test

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
def test_user_orders(db_config):
    do_dataset_test(db_config, "user_orders")


@pytest.mark.parametrize("db_config", DATABASE_CONFIGURATIONS, indirect=True)
def test_data_types(db_config):
    do_dataset_test(db_config, "data_types")


@pytest.mark.parametrize("db_config", DATABASE_CONFIGURATIONS, indirect=True)
def test_fk_chain(db_config):
    do_dataset_test(db_config, "fk_chain")
