from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from subsetter.common import DatabaseConfig, SQLKnownOperator, SQLLiteralType
from subsetter.filters import FilterConfig


class ForbidBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannerConfig(ForbidBaseModel):
    class TargetConfig(ForbidBaseModel):
        all_: bool = Field(False, alias="all")
        percent: Optional[float] = None
        amount: Optional[int] = None
        like: Dict[str, List[str]] = {}
        in_: Dict[str, List[SQLLiteralType]] = Field({}, alias="in")
        sql: Optional[str] = None

    class IgnoreFKConfig(ForbidBaseModel):
        src_table: str
        dst_table: str

    class ExtraFKConfig(ForbidBaseModel):
        src_table: str
        src_columns: List[str]
        dst_table: str
        dst_columns: List[str]

    class ColumnConstraint(ForbidBaseModel):
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
    infer_foreign_keys: Literal["none", "schema", "all"] = "none"
    include_dependencies: bool = True


class DirectoryOutputConfig(ForbidBaseModel):
    mode: Literal["directory"]
    directory: str


ConflictStrategy = Literal["error", "replace", "skip"]


class DatabaseOutputConfig(DatabaseConfig):
    class TableRemapPattern(ForbidBaseModel):
        search: str
        replace: str

    mode: Literal["database"]
    remap: List[TableRemapPattern] = []
    conflict_strategy: ConflictStrategy = "error"
    merge: bool = False


OutputType = Annotated[
    Union[DirectoryOutputConfig, DatabaseOutputConfig],
    Field(..., discriminator="mode"),
]


class SamplerConfig(ForbidBaseModel):
    class MultiplicityConfig(ForbidBaseModel):
        multiplier: int = 1
        ignore_tables: List[str] = []
        extra_columns: Dict[str, List[str]] = {}
        ignore_primary_key_columns: Dict[str, List[str]] = {}

    output: OutputType = DirectoryOutputConfig(mode="directory", directory="output")
    filters: Dict[str, List[FilterConfig]] = {}  # type: ignore
    multiplicity: MultiplicityConfig = MultiplicityConfig()
    infer_foreign_keys: Literal["none", "schema", "all"] = "none"


class SubsetterConfig(ForbidBaseModel):
    source: DatabaseConfig = DatabaseConfig()
    planner: Optional[PlannerConfig] = None
    sampler: Optional[SamplerConfig] = None
