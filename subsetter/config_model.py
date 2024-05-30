from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from subsetter.common import DatabaseConfig, SQLKnownOperator, SQLLiteralType
from subsetter.filters import FilterConfig


class PlannerConfig(BaseModel):
    class TargetConfig(BaseModel):
        all_: bool = Field(False, alias="all")
        percent: Optional[float] = None
        amount: Optional[int] = None
        like: Dict[str, List[str]] = {}
        in_: Dict[str, List[SQLLiteralType]] = Field({}, alias="in")
        sql: Optional[str] = None

    class IgnoreFKConfig(BaseModel):
        src_table: str
        dst_table: str

    class ExtraFKConfig(BaseModel):
        src_table: str
        src_columns: List[str]
        dst_table: str
        dst_columns: List[str]

    class ColumnConstraint(BaseModel):
        column: str
        operator: SQLKnownOperator
        value: Union[SQLLiteralType, List[SQLLiteralType]]

    table_constraints: Dict[str, List[ColumnConstraint]] = {}
    select: List[str]
    targets: Dict[str, TargetConfig]
    ignore: List[str] = []
    passthrough: List[str] = []
    ignore_fks: List[IgnoreFKConfig] = []
    extra_fks: List[ExtraFKConfig] = []
    infer_foreign_keys: bool = False


class DirectoryOutputConfig(BaseModel):
    mode: Literal["directory"]
    directory: str


class DatabaseOutputConfig(DatabaseConfig):
    class TableRemapPattern(BaseModel):
        search: str
        replace: str

    mode: Literal["database"]
    remap: List[TableRemapPattern] = []


OutputType = Annotated[
    Union[DirectoryOutputConfig, DatabaseOutputConfig],
    Field(..., discriminator="mode"),
]


class SamplerConfig(BaseModel):
    class MultiplicityConfig(BaseModel):
        multiplier: int = 1
        infer_foreign_keys: bool = False
        passthrough: List[str] = []
        extra_columns: Dict[str, List[str]] = {}
        ignore_primary_key_columns: Dict[str, List[str]] = {}

    output: OutputType = DirectoryOutputConfig(mode="directory", directory="output")
    filters: Dict[str, List[FilterConfig]] = {}  # type: ignore
    multiplicity: MultiplicityConfig = MultiplicityConfig()


class SubsetterConfig(BaseModel):
    source: DatabaseConfig = DatabaseConfig()
    planner: Optional[PlannerConfig] = None
    sampler: Optional[SamplerConfig] = None
