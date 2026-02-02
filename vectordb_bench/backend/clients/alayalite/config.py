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
    metric_type: MetricType = MetricType.L2
    quantization_type: str = "none"

    # HNSW 常见参数
    M: int = 16
    ef_construction: int = 200
    ef: int = 200

    # ✅ 新增：capacity（关键）
    # 500K 建议至少 600K，留余量避免顶满
    capacity: int = 600000

    # ✅ 可选：max_nbrs 对应 IndexParams.max_nbrs（你查到默认 32）
    # 如果你不想动它可以不加；加了方便对齐 M
    max_nbrs: int = 32

    def index_param(self) -> dict:
        # 目前我们会在 client 的“首次建索引”里用到 capacity/max_nbrs/metric
        return {}

    def search_param(self) -> dict:
        return {"ef_search": self.ef}

