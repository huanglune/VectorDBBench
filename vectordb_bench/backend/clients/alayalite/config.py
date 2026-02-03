from pydantic import BaseModel, validator

from ..api import DBConfig, DBCaseConfig, MetricType


class AlayaLiteConfig(DBConfig):
    url: str = ".alayalite_data"

    def to_dict(self) -> dict:
        return {"url": self.url}

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if field.name in cls.common_short_configs() or field.name in cls.common_long_configs():
            return v
        if isinstance(v, str) and v == "":
            raise ValueError("Empty string!")
        return v


class AlayaLiteHNSWConfig(BaseModel, DBCaseConfig):
    # ✅ 必须保留：给 vdbbench assembler 写入 “数据集语义 metric”
    # assemble 时会被设置成 OpenAI 数据集的 COSINE
    metric_type: MetricType = MetricType.COSINE

    # ✅ 你要控制的：索引使用的 metric（l2/ip/cosine）
    index_metric_type: str = "l2"

    quantization_type: str = "none"

    M: int = 16
    ef_construction: int = 200
    ef: int = 200

    capacity: int = 600000
    max_nbrs: int = 32

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {"ef_search": self.ef}

