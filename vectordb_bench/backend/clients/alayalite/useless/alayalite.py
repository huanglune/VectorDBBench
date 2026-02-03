import logging
from contextlib import contextmanager
from typing import Any

import alayalite

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)


class AlayaLite(VectorDB):
    """
    VectorDBBench client for AlayaLite.

    Goals of this version:
    - Save to disk every N inserted vectors (default: 1000)
    - Print a clearer progress line on each save:
        [AlayaLite][SAVE #<n>] inserted=<total> (+<delta>) save_every=<N> collection=<name>
    - Keep VectorDBBench interface unchanged.
    """

    _SAVE_EVERY_N = 5000  # save every N inserted vectors

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

        # Progress counters
        self._since_last_save = 0          # inserted since last save
        self._inserted_total = 0           # total inserted in this init() lifecycle
        self._save_count = 0               # number of save_collection() calls
        self._save_every = self._SAVE_EVERY_N

        # drop_old: delete disk data for the collection if exists
        if drop_old:
            tmp = alayalite.Client(**self._db_config)
            try:
                if self._collection_name in tmp.list_collections():
                    tmp.delete_collection(collection_name=self._collection_name, delete_on_disk=True)
            finally:
                tmp = None

    def _print_save_progress(self, delta_inserted: int, reason: str) -> None:
        # reason: "periodic" or "final"
        print(
            f"[AlayaLite][SAVE #{self._save_count:03d}] "
            f"inserted={self._inserted_total} (+{delta_inserted}) "
            f"save_every={self._save_every} "
            f"collection={self._collection_name} "
            f"reason={reason}"
        )

    def _save_with_count(self, delta_inserted: int, reason: str) -> None:
        """
        Wrap save_collection with:
        - save counter
        - clearer progress output
        """
        assert self._client is not None, "Please call self.init() before"
        self._client.save_collection(self._collection_name)
        self._save_count += 1
        self._print_save_progress(delta_inserted=delta_inserted, reason=reason)

    @contextmanager
    def init(self) -> None:  # type: ignore
        """
        Benchmark will call: with db.init(): then insert/search/optimize
        """
        try:
            self._client = alayalite.Client(**self._db_config)
            self._collection = self._client.get_or_create_collection(self._collection_name)

            # reset counters per init lifecycle
            self._since_last_save = 0
            self._inserted_total = 0
            self._save_count = 0

            yield

        finally:
            # Final save to persist the last chunk (< save_every)
            try:
                if self._client is not None and self._collection is not None:
                    # delta here is whatever left since last save
                    delta = self._since_last_save
                    self._save_with_count(delta_inserted=delta, reason="final")
                    self._since_last_save = 0
            except Exception as e:
                log.warning(f"Final save_collection failed: {e}")

            self._client = None
            self._collection = None

    def need_normalize_cosine(self) -> bool:
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        """
        Insert embeddings. Save to disk every `save_every` inserted vectors.
        """
        assert self._collection is not None, "Please call self.init() before"
        assert self._client is not None, "Please call self.init() before"
        assert len(embeddings) == len(metadata)

        size = len(embeddings)

        # AlayaLite Collection.insert expects items:
        #   (id, document, embedding, metadata_dict)
        items = []
        for vec, mid in zip(embeddings, metadata):
            items.append((mid, "", vec, {"id": mid}))

        self._collection.insert(items)

        # update counters
        self._inserted_total += size
        self._since_last_save += size

        # periodic save
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
        Return top-k ids as list[int], per VectorDBBench api contract.

        Note: AlayaLite asserts ef_search > limit (k). Ensure this holds.
        """
        assert self._collection is not None, "Please call self.init() before"

        search_kwargs = self._case_config.search_param() if self._case_config else {}

        # Ensure ef_search > k (AlayaLite requirement)
        ef_search = int(search_kwargs.get("ef_search", max(200, k + 1)))
        if ef_search <= k:
            ef_search = max(2 * k, k + 1)
        search_kwargs["ef_search"] = ef_search

        res = self._collection.batch_query([query], limit=k, **search_kwargs)
        return res["id"][0]

    def optimize(self, data_size: int | None = None):
        # AlayaLite builds index on first insert; no additional optimize step needed here.
        return
