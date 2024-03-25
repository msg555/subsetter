import abc
import dataclasses
import string
from datetime import datetime
from fnmatch import fnmatch
from importlib import import_module
from random import Random
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import uuid4

from faker import Faker
from pydantic import BaseModel, Field
from typing_extensions import Annotated

EPOCH = datetime(1970, 1, 1, 0, 0)


@dataclasses.dataclass(frozen=True)
class FilterContext:
    random: Optional[Random] = None
    faker: Optional[Faker] = None


class FilterView(abc.ABC):
    def __init__(
        self, columns_in: Iterable[str], *, columns_out: Optional[Iterable[str]] = None
    ) -> None:
        self.columns_in = tuple(columns_in)
        self.columns_out = (
            self.columns_in if columns_out is None else tuple(columns_out)
        )

    @abc.abstractmethod
    def filter_view(self, values: Iterable[Any]) -> List[Any]:
        pass


def _validate_and_index(
    filter_name: str, columns_in: Tuple[str, ...], columns_act: Tuple[str, ...]
) -> Tuple[int, ...]:
    if not columns_act:
        raise ValueError(f"Filter {filter_name} must act on at least one column")

    cols_seen: Dict[str, int] = {}
    for index, column in enumerate(columns_act):
        if cols_seen.setdefault(column, index) != index:
            raise ValueError(f"Filter {filter_name} has duplicate column {column}")

    column_index_map = {column: index for index, column in enumerate(columns_in)}
    column_act_indices = []
    for column in columns_act:
        act_index = column_index_map.get(column)
        if act_index is None:
            raise ValueError(
                f"Filter {filter_name} refers to non-existant column {column}"
            )
        column_act_indices.append(act_index)
    return tuple(column_act_indices)


class FilterColumns(FilterView):
    def __init__(self, columns_in: Iterable[str], columns_act: Iterable[str]) -> None:
        super().__init__(columns_in)

        self.columns_act = tuple(columns_act)
        self.column_act_indices = _validate_and_index(
            str(type(self)), self.columns_in, self.columns_act
        )

    def filter_view(self, values: Iterable[Any]) -> List[Any]:
        values_out = list(values)
        filtered_values = self.filter_values(
            [values_out[index] for index in self.column_act_indices]
        )
        for index, filtered_value in zip(self.column_act_indices, filtered_values):
            values_out[index] = filtered_value
        return values_out

    @abc.abstractmethod
    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        pass


class FilterViewChain(FilterView):
    def __init__(
        self,
        columns_in: Iterable[str],
        columns_out: Iterable[str],
        filter_views: Iterable[FilterView],
    ) -> None:
        super().__init__(columns_in, columns_out=columns_out)
        self.filter_views = filter_views

    def filter_view(self, values: Iterable[Any]) -> List[Any]:
        result = list(values)
        for filter_view in self.filter_views:
            result = filter_view.filter_view(result)
        return result

    @classmethod
    def construct_filter(
        cls,
        columns_in: Iterable[str],
        filter_configs: Iterable["FilterConfig"],  # type: ignore
        *,
        filter_context=FilterContext(),
    ) -> "FilterViewChain":
        cols_in = tuple(columns_in)
        cols_out = cols_in
        filter_views = []
        for filter_config in filter_configs:
            filter_view = filter_config.construct_filter(cols_out, filter_context=filter_context)  # type: ignore
            cols_out = filter_view.columns_out
            filter_views.append(filter_view)
        return cls(cols_in, cols_out, filter_views)


class SingleFilterConfig(BaseModel, abc.ABC):
    @abc.abstractmethod
    def construct_filter(
        self, columns_in: Iterable[str], *, filter_context=FilterContext()
    ) -> FilterView:
        pass


class FilterZero(FilterColumns):
    class Config(SingleFilterConfig):
        op: Literal["zero"]
        columns: List[str]

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            return FilterZero(columns_in, self.columns)

    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        for value in values:
            if value is None:
                yield None
            elif isinstance(value, str):
                yield ""
            elif isinstance(value, int):
                yield 0
            elif isinstance(value, float):
                yield 0.0
            elif isinstance(value, datetime):
                yield EPOCH.replace(tzinfo=value.tzinfo)
            else:
                raise ValueError(f"Cannot zero unknown type {str(type(value))}")


class FilterOmit(FilterView):
    class Config(SingleFilterConfig):
        op: Literal["omit"]
        columns: List[str]

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            return FilterOmit(columns_in, self.columns)

    def __init__(self, columns_in: Iterable[str], columns_act: Iterable[str]) -> None:
        columns_in = tuple(columns_in)
        self.columns_act = tuple(columns_act)
        column_act_set = set(self.columns_act)
        columns_out = tuple(
            column for column in columns_in if column not in column_act_set
        )
        super().__init__(columns_in, columns_out=columns_out)
        columns_in_index = {column: index for index, column in enumerate(columns_in)}
        self.columns_out_indices = [columns_in_index[column] for column in columns_out]

    def filter_view(self, values: Iterable[Any]) -> List[Any]:
        values_in = list(values)
        return [values_in[index] for index in self.columns_out_indices]


class FilterNull(FilterColumns):
    class Config(SingleFilterConfig):
        op: Literal["null"]
        columns: List[str]

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            return FilterNull(columns_in, self.columns)

    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        return (None for _ in values)


class FilterRandomInt(FilterColumns):
    class Config(SingleFilterConfig):
        op: Literal["random_int"]
        columns: List[str]
        low: int
        high: int

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            return FilterRandomInt(
                columns_in, self.columns, self.low, self.high, rng=filter_context.random
            )

    def __init__(
        self,
        columns_in: Iterable[str],
        columns_act: Iterable[str],
        low: int,
        high: int,
        *,
        rng: Optional[Random] = None,
    ) -> None:
        super().__init__(columns_in, columns_act)
        self.low = low
        self.high = high
        self.rng = rng or Random()

    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        return (self.rng.randint(self.low, self.high) for _ in values)


class FilterRandomFloat(FilterColumns):
    class Config(SingleFilterConfig):
        op: Literal["random_float"]
        columns: List[str]
        low: float
        high: float

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            return FilterRandomFloat(
                columns_in, self.columns, self.low, self.high, rng=filter_context.random
            )

    def __init__(
        self,
        columns_in: Iterable[str],
        columns_act: Iterable[str],
        low: float,
        high: float,
        *,
        rng: Optional[Random] = None,
    ) -> None:
        super().__init__(columns_in, columns_act)
        self.low = low
        self.high = high
        self.rng = rng or Random()

    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        return (self.rng.uniform(self.low, self.high) for _ in values)


_ALPHABET_MAP = {
    "alnum": string.ascii_letters + string.digits,
    "hex": string.digits + "abcdef",
    "hex_lower": string.digits + "abcdef",
    "hex_upper": string.digits + "ABCDEF",
    "digit": string.digits,
    "alpha": string.ascii_letters,
    "alpha_lower": string.ascii_lowercase,
    "alpha_upper": string.ascii_uppercase,
}
_ALPHABET_TYPES = Literal[
    "alnum",
    "hex",
    "hex_lower",
    "hex_upper",
    "digit",
    "alpha",
    "alpha_lower",
    "alpha_upper",
    "custom",
]


class FilterRandomString(FilterColumns):
    class Config(SingleFilterConfig):
        op: Literal["random_string"]
        columns: List[str]
        length: int
        alphabet: _ALPHABET_TYPES = "alnum"
        custom_alphabet: Optional[str] = None

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            alphabet = _ALPHABET_MAP.get(self.alphabet, self.custom_alphabet)
            if not alphabet:
                raise ValueError("Custom alphabet cannot be empty")
            return FilterRandomString(
                columns_in,
                self.columns,
                self.length,
                alphabet,
                rng=filter_context.random,
            )

    def __init__(
        self,
        columns_in: Iterable[str],
        columns_act: Iterable[str],
        length: int,
        alphabet: str,
        *,
        rng: Optional[Random] = None,
    ) -> None:
        super().__init__(columns_in, columns_act)
        self.length = length
        self.alphabet = alphabet
        self.rng = rng or Random()

    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        return ("".join(self.rng.choices(self.alphabet, k=self.length)) for _ in values)


FilterConstantTypes = Union[None, float, int, str]


class FilterConstant(FilterColumns):
    class Config(SingleFilterConfig):
        op: Literal["constant"]
        columns: List[str]
        values: List[FilterConstantTypes]

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            return FilterConstant(columns_in, self.columns, self.values)

    def __init__(
        self,
        columns_in: Iterable[str],
        columns_act: Iterable[str],
        values: Iterable[FilterConstantTypes],
    ) -> None:
        super().__init__(columns_in, columns_act)
        self.values = tuple(values)
        if len(self.values) != len(self.columns_act):
            raise ValueError(
                "Number of columns and values must match for constant filter"
            )

    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        return self.values


class FilterUuid(FilterColumns):
    class Config(SingleFilterConfig):
        op: Literal["uuid"]
        columns: List[str]

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            return FilterUuid(columns_in, self.columns)

    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        return (str(uuid4()) for _ in values)


class FilterFakeColumns(FilterColumns):
    def __init__(
        self,
        columns_in: Iterable[str],
        columns_act: Iterable[str],
        *,
        unique: bool = False,
        faker: Optional[Faker] = None,
    ):
        super().__init__(columns_in, columns_act)
        faker = faker or Faker()
        self.faker: Faker = faker.unique if unique else faker  # type: ignore


FAKER_CONFIG_CLASSES = []


def _simple_faker_class(op_name: str, faker_op: Callable[[Faker], Any]):
    class FilterFakeOper(FilterFakeColumns):
        class Config(SingleFilterConfig):
            op: Literal[op_name]  # type: ignore
            columns: List[str]
            unique: bool = False

            def construct_filter(
                self, columns_in: Iterable[str], *, filter_context=FilterContext()
            ) -> FilterView:
                return FilterFakeOper(
                    columns_in,
                    self.columns,
                    unique=self.unique,
                    faker=filter_context.faker,
                )

        def filter_values(self, values: List[Any]) -> Iterable[Any]:
            return (faker_op(self.faker) for _ in values)

    name = "".join(part.title() for part in op_name.split("_"))
    FilterFakeOper.__name__ = f"Filter{name}"
    FilterFakeOper.__qualname__ = f"Filter{name}"
    FilterFakeOper.Config.__name__ = f"Filter{name}.Config"
    FilterFakeOper.Config.__qualname__ = f"Filter{name}.Config"
    FAKER_CONFIG_CLASSES.append(FilterFakeOper.Config)
    return FilterFakeOper


FilterFakeEmail = _simple_faker_class("fake_email", lambda faker: faker.email())
FilterFakeFirstName = _simple_faker_class(
    "fake_first_name", lambda faker: faker.first_name()
)
FilterFakeLastName = _simple_faker_class(
    "fake_last_name", lambda faker: faker.last_name()
)
FilterFakeName = _simple_faker_class("fake_name", lambda faker: faker.name())
FilterFakePhoneNumber = _simple_faker_class(
    "fake_phone_number", lambda faker: faker.phone_number()
)
FilterFakeLicensePlate = _simple_faker_class(
    "fake_license_plate", lambda faker: faker.licence_plate()
)
FilterFakeVin = _simple_faker_class("fake_vin", lambda faker: faker.vin())
FilterFakeAddress = _simple_faker_class("fake_address", lambda faker: faker.address())
FilterFakeBuildingNumber = _simple_faker_class(
    "fake_building_number", lambda faker: faker.building_number()
)
FilterFakeCity = _simple_faker_class("fake_city", lambda faker: faker.city())
FilterFakeState = _simple_faker_class("fake_state", lambda faker: faker.state())
FilterFakeStateAbbr = _simple_faker_class(
    "fake_state_abbr", lambda faker: faker.state_abbr()
)
FilterFakeCountry = _simple_faker_class("fake_country", lambda faker: faker.country())
FilterFakeCountryCode = _simple_faker_class(
    "fake_country_code", lambda faker: faker.country_code()
)
FilterFakePostalCode = _simple_faker_class(
    "fake_postal_code", lambda faker: faker.postcode()
)
FilterFakeStreetAddress = _simple_faker_class(
    "fake_street_address", lambda faker: faker.street_address()
)
FilterFakeStreetName = _simple_faker_class(
    "fake_street_name", lambda faker: faker.street_name()
)
FilterFakeLatitude = _simple_faker_class(
    "fake_latitude", lambda faker: faker.latitude()
)
FilterFakeLongitude = _simple_faker_class(
    "fake_longitude", lambda faker: faker.longitude()
)


class FilterPlugin(FilterColumns):
    """
    Filter that supports loading a custom plugin by module and class name.
    """

    PluginType = Callable[[List[Any]], Iterable[Any]]

    class Config(SingleFilterConfig):  # type: ignore
        op: Literal["plugin"]
        columns: List[str]

        module: str
        clazz: str = Field(..., alias="class")
        kwargs: Dict[str, Any] = {}

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            try:
                try:
                    module = import_module(self.module)
                except ImportError as exc:
                    raise ValueError(
                        f"Failed to import module {self.module} for plugin"
                    ) from exc
                try:
                    clazz = getattr(module, self.clazz)
                except AttributeError as exc:
                    raise ValueError(
                        f"Could not find plugin attribute {self.clazz} in {self.module}"
                    ) from exc
                plugin = clazz(**self.kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                raise ValueError(
                    f"Uncaught {type(exc).__name__} when attempting to "
                    f"instantiate plugin {self.module}.{self.clazz}: {exc}"
                ) from exc

            return FilterPlugin(columns_in, self.columns, plugin)

    def __init__(
        self,
        columns_in: Iterable[str],
        columns_act: Iterable[str],
        plugin: PluginType,
    ) -> None:
        super().__init__(columns_in, columns_act)
        self.plugin = plugin

    def filter_values(self, values: List[Any]) -> Iterable[Any]:
        return self.plugin(values)


SimpleFilterConfig: Type[SingleFilterConfig] = Annotated[  # type: ignore
    Union[
        FilterZero.Config,
        FilterOmit.Config,
        FilterNull.Config,
        FilterConstant.Config,
        FilterRandomInt.Config,
        FilterRandomFloat.Config,
        FilterRandomString.Config,
        FilterUuid.Config,
        FilterFakeEmail.Config,
        FilterFakeFirstName.Config,
        FilterFakeLastName.Config,
        FilterFakeName.Config,
        FilterFakePhoneNumber.Config,
        FilterFakeLicensePlate.Config,
        FilterFakeVin.Config,
        FilterFakeAddress.Config,
        FilterFakeBuildingNumber.Config,
        FilterFakeCity.Config,
        FilterFakeState.Config,
        FilterFakeStateAbbr.Config,
        FilterFakeCountry.Config,
        FilterFakeCountryCode.Config,
        FilterFakePostalCode.Config,
        FilterFakeStreetAddress.Config,
        FilterFakeStreetName.Config,
        FilterFakeLatitude.Config,
        FilterFakeLongitude.Config,
        FilterPlugin.Config,
    ],
    Field(..., discriminator="op"),
]


class MultiplexCondition(BaseModel):
    op: Literal[
        "is", "is not", "in", "not in", "=", "!=", "<", ">", "<=", ">=", "matches"
    ]
    value: Union[None, int, float, str, List[Union[None, int, float, str]]]
    apply: SimpleFilterConfig  # type: ignore

    def match(self, value: Any) -> bool:
        op = self.op
        if op == "is":
            return value is self.value
        if op == "is not":
            return value is not self.value
        if op == "in":
            return value in self.value  # type: ignore
        if op == "not in":
            return value not in self.value  # type: ignore
        if op == "=":
            return value == self.value
        if op == "!=":
            return value != self.value
        if op == "<":
            return value < self.value
        if op == ">":
            return value > self.value
        if op == "<=":
            return value <= self.value
        if op == ">=":
            return value >= self.value
        if op == "matches":
            return fnmatch(value, self.value)
        raise ValueError("Unknown op")


class FilterMultiplex(FilterView):
    class Config(SingleFilterConfig):
        op: Literal["multiplex"]
        column: str
        conditions: List[MultiplexCondition]

        def construct_filter(
            self, columns_in: Iterable[str], *, filter_context=FilterContext()
        ) -> FilterView:
            return FilterMultiplex(
                columns_in,
                self.op,
                self.column,
                self.conditions,
                filter_context=filter_context,
            )

    def __init__(
        self,
        columns_in: Iterable[str],
        op: str,
        column: str,
        conditions: Iterable[MultiplexCondition],
        *,
        filter_context=FilterContext(),
    ) -> None:
        super().__init__(columns_in)
        self.op = op
        self.column = column
        try:
            self.column_index = self.columns_in.index(column)
        except ValueError as exc:
            raise ValueError("Multiplex column does not exist") from exc
        self.conditions = tuple(conditions)
        self.condition_filters = tuple(
            condition.apply.construct_filter(columns_in, filter_context=filter_context)  # type: ignore
            for condition in self.conditions
        )
        for condition_filter in self.condition_filters:
            if condition_filter.columns_out != self.columns_out:
                raise ValueError(
                    "Cannot use filters with multiplex that modify columns"
                )

    def filter_view(self, values: Iterable[Any]) -> List[Any]:
        values_in = tuple(values)
        value = values_in[self.column_index]
        for condition, condition_filter in zip(self.conditions, self.condition_filters):
            if condition.match(value):
                return condition_filter.filter_view(values_in)
        return list(values_in)


FilterConfig: Type[SingleFilterConfig] = Annotated[  # type: ignore
    Union[
        SimpleFilterConfig,
        FilterMultiplex.Config,
    ],
    Field(..., discriminator="op"),
]
