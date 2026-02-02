# vectordb_bench/backend/clients/alayalite/alayalite.py

import logging
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd

import alayalite
from alayalite.index import Index
from alayalite.schema import IndexParams as _IndexParams

from ..api import DBCaseConfig, VectorDB, contextmanager

log = logging.getLogger(__name__)

# --------------------------
# Monkeypatch: make IndexParams.to_json_dict JSON-serializable
# (fixes: "Object of type uint32 is not JSON serializable")
# --------------------------
_PATCHED_JSON = False


def _jsonable(obj):
    # numpy scalar -> python scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # dict/list recursive
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _patch_indexparams_to_json_dict():
    global _PATCHED_JSON
    if _PATCHED_JSON:
        return
    _PATCHED_JSON = True

    if hasattr(_IndexParams, "to_json_dict"):
        _orig = _IndexParams.to_json_dict

        def _wrapped(self):  # type: ignore
            d = _orig(self)
            return _jsonable(d)

        _IndexParams.to_json_dict = _wrapped  # type: ignore


_patch_indexparams_to_json_dict()


class AlayaLite(VectorDB):
    """
    VectorDBBench client for AlayaLite.

    Key behaviors:
    - Robust drop_old: deletes collection directory without requiring schema parsing.
    - Bootstraps the first index with capacity injected (avoids default capacity=100000).
    - Periodic + final save with clear progress prints.
    - Ensures ef_search > k to satisfy alayalite assertions.
    """

    # IO-friendly default; you can change this constant if needed
    _DEFAULT_SAVE_EVERY = 5000

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        name: str = "AlayaLite",
        **kwargs: Any,
    ) -> None:
        log.warning(f"AlayaLite init, drop old: {drop_old}")

        self.name = name
        self._dim = dim
        self._db_config = db_config
        self._case_config = db_case_config
        self._collection_name = collection_name
        self._drop_old = drop_old

        self._client = None
        self._collection = None

        # Save counters
        self._since_last_save = 0
        self._inserted_total = 0
        self._save_count = 0

        # Save frequency (use config if provided; else default)
        self._save_every = int(getattr(db_case_config, "save_every", self._DEFAULT_SAVE_EVERY))

        # Capacity for index (critical for >100K)
        self._capacity = int(getattr(db_case_config, "capacity", 100000))

        # Optional neighbors cap (matches IndexParams.max_nbrs default 32)
        self._max_nbrs = int(getattr(db_case_config, "max_nbrs", 32))

        # Bootstrap flag: whether we created index manually with capacity
        self._bootstrapped = False

        # Robust drop_old: delete collection directory directly (works even if schema.json is corrupted)
        if drop_old:
            url = self._db_config.get("url")
            if isinstance(url, str) and url:
                abs_url = os.path.abspath(url)
                col_dir = os.path.join(abs_url, self._collection_name)
                if os.path.exists(col_dir):
                    shutil.rmtree(col_dir)
                    print(f"Collection {self._collection_name} is deleted")

    def need_normalize_cosine(self) -> bool:
        # Keep as False (matches your earlier working runs)
        return True

    def _print_save_progress(self, delta_inserted: int, reason: str) -> None:
        print(
            f"[AlayaLite][SAVE #{self._save_count:03d}] "
            f"inserted={self._inserted_total} (+{delta_inserted}) "
            f"save_every={self._save_every} "
            f"collection={self._collection_name} "
            f"reason={reason}"
        )

    def _save_with_count(self, delta_inserted: int, reason: str) -> None:
        assert self._client is not None, "Please call self.init() before"
        self._client.save_collection(self._collection_name)  # alayalite itself prints "Collection ... is saved"
        self._save_count += 1
        self._print_save_progress(delta_inserted=delta_inserted, reason=reason)

    def _metric_to_str(self) -> str:
        # alayalite 后端不支持 COSINE enum，因此禁止使用 cosine
        # 统一用 l2（配合 need_normalize_cosine=True）
        return "l2"


    def _bootstrap_index_with_capacity(self, embeddings: list[list[float]], metadata: list[int]) -> None:
        """
        Build the index once with capacity injected, then inject it into Collection.

        This bypasses alayalite.Collection.insert()'s first-branch default capacity=100000.
        """
        assert self._collection is not None, "Please call self.init() before"

        vecs = np.array(embeddings, dtype=np.float32)

        params = _IndexParams(
            data_type=vecs.dtype,
            metric=self._metric_to_str(),
            capacity=int(max(self._capacity, len(vecs))),  # IMPORTANT: keep as python int
            max_nbrs=int(self._max_nbrs),
        )
        index = Index(self._collection_name, params)

        efc = int(getattr(self._case_config, "ef_construction", 100)) if self._case_config else 100
        index.fit(vecs, ef_construction=efc, num_threads=1)

        # Inject index into Collection private field
        setattr(self._collection, "_Collection__index_py", index)

        # Populate dataframe and id mappings to match alayalite.Collection.insert(first-branch)
        df = getattr(self._collection, "_Collection__dataframe")
        outer_inner = getattr(self._collection, "_Collection__outer_inner_map")
        inner_outer = getattr(self._collection, "_Collection__inner_outer_map")

        new_rows = []
        for i, mid in enumerate(metadata):
            new_rows.append({"id": mid, "document": "", "metadata": {"id": mid}})
            outer_inner[mid] = i
            inner_outer[i] = mid

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        setattr(self._collection, "_Collection__dataframe", df)

        self._bootstrapped = True

    @contextmanager
    def init(self) -> None:  # type: ignore
        try:
            self._client = alayalite.Client(**self._db_config)
            self._collection = self._client.get_or_create_collection(self._collection_name)

            self._since_last_save = 0
            self._inserted_total = 0
            self._save_count = 0
            self._bootstrapped = False

            yield

        finally:
            try:
                if self._client is not None and self._collection is not None:
                    # 只有 index 已建立才保存，避免 __index_py 为 None 时 save 崩
                    if getattr(self._collection, "_Collection__index_py", None) is not None:
                        delta = self._since_last_save
                        self._save_with_count(delta_inserted=delta, reason="final")
                        self._since_last_save = 0

            except Exception as e:
                log.warning(f"Final save_collection failed: {e}")

            self._client = None
            self._collection = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        """
        VectorDBBench calls this in batches.

        metadata is a list of ids (ints). We store them as:
        - id = metadata value
        - document = ""
        - metadata dict = {"id": metadata value}
        """
        assert self._collection is not None, "Please call self.init() before"
        assert self._client is not None, "Please call self.init() before"
        assert len(embeddings) == len(metadata)

        size = len(embeddings)

        # First batch: bootstrap index with injected capacity
        if (not self._bootstrapped) and getattr(self._collection, "_Collection__index_py", None) is None:
            self._bootstrap_index_with_capacity(embeddings, metadata)
        else:
            items = [(mid, "", vec, {"id": mid}) for vec, mid in zip(embeddings, metadata)]
            self._collection.insert(items)

        # Counters
        self._inserted_total += size
        self._since_last_save += size

        # Periodic save
        if self._since_last_save >= self._save_every:
            delta = self._since_last_save
            self._save_with_count(delta_inserted=delta, reason="periodic")
            self._since_last_save = 0

        return size, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """
        VectorDBBench search API.
        """
        assert self._collection is not None, "Please call self.init() before"

        search_kwargs = self._case_config.search_param() if self._case_config else {}

        # Ensure ef_search strictly > k (alayalite's assertion can be >, not >=)
        ef_search = int(search_kwargs.get("ef_search", max(200, k + 1)))
        if ef_search <= k:
            ef_search = max(2 * k, k + 1)
        search_kwargs["ef_search"] = ef_search

        res = self._collection.batch_query([query], limit=k, **search_kwargs)
        return res["id"][0]

    def optimize(self, data_size: int | None = None):
        # No-op for AlayaLite
        return
