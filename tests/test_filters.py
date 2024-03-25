import string
import typing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import faker

from subsetter.filters import (
    FilterConstant,
    FilterContext,
    FilterFakeAddress,
    FilterFakeBuildingNumber,
    FilterFakeCity,
    FilterFakeCountry,
    FilterFakeCountryCode,
    FilterFakeEmail,
    FilterFakeFirstName,
    FilterFakeLastName,
    FilterFakeLatitude,
    FilterFakeLicensePlate,
    FilterFakeLongitude,
    FilterFakeName,
    FilterFakePhoneNumber,
    FilterFakePostalCode,
    FilterFakeState,
    FilterFakeStateAbbr,
    FilterFakeStreetAddress,
    FilterFakeStreetName,
    FilterFakeVin,
    FilterNull,
    FilterOmit,
    FilterRandomFloat,
    FilterRandomInt,
    FilterRandomString,
    FilterUuid,
    FilterZero,
)

EPOCH_UTC = datetime.fromtimestamp(0, tz=timezone.utc)
EPOCH_NO_TZ = EPOCH_UTC.replace(tzinfo=None)


def test_filter_omit() -> None:
    filt = FilterOmit.Config(
        op="omit",
        columns=["col2", "col4", "col5"],
    ).construct_filter(["col1", "col2", "col3", "col4", "col5"])
    assert filt.filter_view(["val1", "val2", "val3", "val4", "val5"]) == [
        "val1",
        "val3",
    ]


def _column_filter_test(
    filter_cls,
    values_in: List[Any],
    exp_values_out: List[Any],
    *,
    config_extra: Optional[Dict[str, Any]] = None,
    filter_context=FilterContext(),
) -> None:
    columns = [f"col_{i}" for i in range(len(values_in))]
    config_dict = {
        "op": typing.get_args(filter_cls.Config.__annotations__["op"])[0],
        "columns": columns,
        **(config_extra or {}),
    }
    filt = filter_cls.Config.model_validate(config_dict).construct_filter(
        columns, filter_context=filter_context
    )
    values_out = filt.filter_view(values_in)
    for value_out, exp_value_out in zip(values_out, exp_values_out):
        assert value_out == exp_value_out
        assert type(value_out) is type(exp_value_out)


def test_filter_zero() -> None:
    dta = datetime.now(tz=timezone.utc)
    dtb = dta.replace(tzinfo=None)
    _column_filter_test(
        FilterZero,
        ["hello", "", 2, 123.4, None, dta, dtb],
        ["", "", 0, 0.0, None, EPOCH_UTC, EPOCH_NO_TZ],
    )


def test_filter_null() -> None:
    _column_filter_test(
        FilterNull,
        ["hello", "", 2, 123.4, None],
        [None, None, None, None, None],
    )


def test_filter_constant() -> None:
    _column_filter_test(
        FilterConstant,
        ["hello", "", 2, 123.4, "x"],
        [None, 123, 555.5, "wow", "x"],
        config_extra={
            "values": [
                None,
                123,
                555.5,
                "wow",
                "x",
            ]
        },
    )


def test_filter_random_int() -> None:
    exp_values_out = [123, 234, 345, 456]
    mock_rng = Mock()
    mock_rng.randint = Mock(side_effect=exp_values_out)
    _column_filter_test(
        FilterRandomInt,
        [1, 2, 3, 4],
        exp_values_out,
        config_extra={
            "low": 123,
            "high": 555,
        },
        filter_context=FilterContext(random=mock_rng),
    )
    mock_rng.randint.assert_called_with(123, 555)


def test_filter_random_float() -> None:
    exp_values_out = [123.4, 234.5, 345.6, 456.7]
    mock_rng = Mock()
    mock_rng.uniform = Mock(side_effect=exp_values_out)
    _column_filter_test(
        FilterRandomFloat,
        [1.2, 2.3, 3.4, 4.5],
        exp_values_out,
        config_extra={
            "low": 123.4,
            "high": 555.5,
        },
        filter_context=FilterContext(random=mock_rng),
    )
    mock_rng.uniform.assert_called_with(123.4, 555.5)


def test_filter_random_string() -> None:
    exp_values_out = ["qwer", "asdf", "zxcv"]
    mock_rng = Mock()
    mock_rng.choices = Mock(side_effect=exp_values_out)
    _column_filter_test(
        FilterRandomString,
        ["ABC", "DEF", "GHI"],
        exp_values_out,
        config_extra={
            "length": 4,
        },
        filter_context=FilterContext(random=mock_rng),
    )
    mock_rng.choices.assert_called_with(string.ascii_letters + string.digits, k=4)

    filt = FilterRandomString.Config(
        op="random_string",
        columns=["col"],
        length=123,
        alphabet="hex_lower",
    ).construct_filter(["col"])
    assert filt.alphabet == "0123456789abcdef"  # type: ignore

    filt = FilterRandomString.Config(
        op="random_string",
        columns=["col"],
        length=123,
        alphabet="digit",
    ).construct_filter(["col"])
    assert filt.alphabet == "0123456789"  # type: ignore

    filt = FilterRandomString.Config(
        op="random_string",
        columns=["col"],
        length=123,
        alphabet="custom",
        custom_alphabet="wowcool",
    ).construct_filter(["col"])
    assert filt.alphabet == "wowcool"  # type: ignore


def test_filter_uuid() -> None:
    exp_values_out = [
        "ea3c50a7-18e2-4ce2-b713-7983dc2cfe43",
        "230b0280-9c3d-4f19-892e-fbe00123759c",
        "526a2846-8dda-4723-a5ae-a622f9b28071",
    ]
    with patch("subsetter.filters.uuid4", side_effect=exp_values_out):
        _column_filter_test(
            FilterUuid,
            [None, None, None],
            exp_values_out,
        )


def test_fake_filters() -> None:
    filters = [
        (FilterFakeEmail, "email", "abc@def.com"),
        (FilterFakeFirstName, "first_name", "John"),
        (FilterFakeLastName, "last_name", "Smith"),
        (FilterFakeName, "name", "John Smith"),
        (FilterFakePhoneNumber, "phone_number", "123-456-7899"),
        (FilterFakeLicensePlate, "licence_plate", "123-XYAB"),
        (FilterFakeVin, "vin", "5FNRL384X7B133819"),
        (FilterFakeAddress, "address", "123 Easy St, The Moon, 99999"),
        (FilterFakeBuildingNumber, "building_number", 1234),
        (FilterFakeCity, "city", "Shire"),
        (FilterFakeState, "state", "Michigan"),
        (FilterFakeStateAbbr, "state_abbr", "MI"),
        (FilterFakeCountry, "country", "Canada"),
        (FilterFakeCountryCode, "country_code", "CA"),
        (FilterFakePostalCode, "postcode", 99999),
        (FilterFakeStreetAddress, "street_address", "123 Easy St"),
        (FilterFakeStreetName, "street_name", "Easy St"),
        (FilterFakeLatitude, "latitude", 44.44),
        (FilterFakeLongitude, "longitude", 33.33),
    ]
    for filter_cls, faker_method, fake_value in filters:
        mock_faker = Mock()
        setattr(mock_faker, faker_method, Mock(return_value=fake_value))
        filter_context = FilterContext(faker=mock_faker)
        _column_filter_test(
            filter_cls, [None], [fake_value], filter_context=filter_context
        )
        getattr(mock_faker, faker_method).assert_called_once()


def test_fake_unique_filters() -> None:
    fake = faker.Faker(locale="en_US")
    filt = FilterFakeStateAbbr(["col"], ["col"], unique=True, faker=fake)

    states = set()
    for _ in range(100):
        try:
            (state,) = filt.filter_view(["abc"])
            states.add(state)
        except faker.exceptions.UniquenessException:
            break
    else:
        assert False, "expected to hit uniqueness exception"

    # Tolerate some political instability
    assert 40 <= len(states) <= 60
