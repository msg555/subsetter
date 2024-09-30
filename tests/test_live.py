import os
import tempfile

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


@pytest.fixture
def sqlite_init_db():
    with tempfile.NamedTemporaryFile(suffix=".db") as tf1:
        with tempfile.NamedTemporaryFile(suffix=".db") as tf2:
            yield tf1.name, tf2.name


def db_config_sqlite(request):
    db1, db2 = request.getfixturevalue("sqlite_init_db")
    return DatabaseConfig(
        dialect="sqlite",
        sqlite_databases={
            "test": db1,
            "test_out": db2,
        },
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
    pytest.param(
        db_config_sqlite,
        marks=[
            pytest.mark.usefixtures("sqlite_init_db"),
            pytest.mark.sqlite_live,
        ],
        id="sqlite",
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


@pytest.mark.parametrize("db_config", DATABASE_CONFIGURATIONS, indirect=True)
def test_instruments(db_config):
    do_dataset_test(db_config, "instruments")
