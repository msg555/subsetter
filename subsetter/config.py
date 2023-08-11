from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class TargetConfig(BaseModel):
    percent: Optional[float] = None
    amount: Optional[int] = None


class IgnoreFKConfig(BaseModel):
    src_table: str
    dst_table: str


class ExtraFKConfig(BaseModel):
    src_table: str
    src_columns: List[str]
    dst_table: str
    dst_columns: List[str]


class SourceConfig(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None


class ColumnConstraint(BaseModel):
    column: str
    operator: Literal["<", ">", "=", "<>", "!=", "in", "not in"]
    expression: Union[float, str, List[Union[float, str]]]


class DirectoryOutputConfig(BaseModel):
    mode: Literal["directory"]
    directory: str


class MysqlOutputConfig(BaseModel):
    mode: Literal["mysql"]
    host: Optional[str] = None
    port: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


OutputType = Annotated[
    Union[DirectoryOutputConfig, MysqlOutputConfig],
    Field(..., discriminator="mode"),
]


class SubsetConfig(BaseModel):
    source: SourceConfig = SourceConfig()
    destination: OutputType = DirectoryOutputConfig(
        mode="directory", directory="output"
    )

    global_constraints: List[ColumnConstraint] = []
    table_constraints: Dict[str, List[ColumnConstraint]] = {}

    schemas: List[str]
    targets: Dict[str, TargetConfig]
    ignore: List[str] = []
    passthrough: List[str] = []
    ignore_fks: List[IgnoreFKConfig] = []
    extra_fks: List[ExtraFKConfig] = []
