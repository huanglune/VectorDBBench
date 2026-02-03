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

    metric_type: Annotated[
        str,
        click.option(
            "--metric-type",
            type=click.Choice(["l2", "ip", "cosine"], case_sensitive=False),
            default="l2",
            show_default=True,
            help="Distance metric for AlayaLite index (l2/ip/cosine).",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(AlayaLiteTypedDict)
def AlayaLite(**params: Unpack[AlayaLiteTypedDict]):
    from .config import AlayaLiteConfig, AlayaLiteHNSWConfig

    quant = params.get("quantization", None)
    metric = (params.get("metric_type", "l2") or "l2").strip().lower()

    case_params = {
        "quantization_type": quant,
        # ✅ 关键：写入 index_metric_type
        "index_metric_type": metric,
        "M": params.get("m", None),
        "ef_construction": params.get("ef_construction", None),
        "ef": params.get("ef_search", None),
    }

    run(
        db=DB.AlayaLite,
        db_config=AlayaLiteConfig(url=params["url"]),
        db_case_config=AlayaLiteHNSWConfig(**{k: v for k, v in case_params.items() if v is not None}),
        **params,
    )

