from typing import Annotated, TypedDict, Unpack

import click
import os
from pydantic import SecretStr

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
        str, click.option("--url", type=str, help="uri connection string", required=False)
    ]

@cli.command()
@click_parameter_decorators_from_typed_dict(AlayaLiteTypedDict)
def AlayaLite(**params: Unpack[AlayaLiteTypedDict]):
    from .config import AlayaLiteConfig, AlayaLiteHNSWConfig

    case_params = {
        "quantization_type": params["quantization"],
        "M": params["m"],
        "ef_construction": params["ef_construction"],
        "ef": params["ef_search"],
    }
    run(
        db=DB.AlayaLite,
        db_config=AlayaLiteConfig(url=SecretStr(params["url"])),
        db_case_config=AlayaLiteHNSWConfig(**{k: v for k, v in case_params.items() if v}),
        **params,
    )