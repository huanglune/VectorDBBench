from pydantic import SecretStr, validator, BaseModel
from ..api import DBConfig, DBCaseConfig, IndexType, MetricType

class AlayaLiteConfig(DBConfig):
    url: str = '.alayalite_data'

    def to_dict(self) -> dict:
        return {
            "url": self.url,
        }

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if field.name in cls.common_short_configs() or field.name in cls.common_long_configs():
            return v
        if not v and isinstance(v, str | SecretStr):
            raise ValueError("Empty string!")
        return v

class AlayaLiteHNSWConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.L2
    quantization_type: str = "none"
    ef_construction: int = 200
    ef: int = 100

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {
            "ef_search": self.ef
        }
