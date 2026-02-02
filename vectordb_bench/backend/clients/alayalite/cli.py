from typing import Annotated, TypedDict, Unpack

import click

from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor1,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from vectordb_bench.backend.clients import DB


class AlayaLiteTypedDict(CommonTypedDict, HNSWFlavor1):
    url: Annotated[
        str,
        click.option(
            "--url",
            type=str,
            help="AlayaLite local data directory (will be created if not exists)",
            required=False,
            default=".alayalite_data",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(AlayaLiteTypedDict)
def AlayaLite(**params: Unpack[AlayaLiteTypedDict]):
    from .config import AlayaLiteConfig, AlayaLiteHNSWConfig

    # 关键修复：有些版本的 HNSWFlavor1 根本没有 quantization 参数
    # 所以这里必须用 params.get()，否则 KeyError
    quant = params.get("quantization", None)

    case_params = {
        "quantization_type": quant,
        "M": params.get("m", None),
        "ef_construction": params.get("ef_construction", None),
        "ef": params.get("ef_search", None),
    }

    run(
        db=DB.AlayaLite,
        # 注意：url 传 str，不要 SecretStr
        db_config=AlayaLiteConfig(url=params["url"]),
        # 只传入不为 None 的字段
        db_case_config=AlayaLiteHNSWConfig(**{k: v for k, v in case_params.items() if v is not None}),
        **params,
    )
