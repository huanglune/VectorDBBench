import logging

import alayalite
from ..api import DBCaseConfig, VectorDB, contextmanager
from .config import AlayaLiteHNSWConfig
log = logging.getLogger(__name__)

class AlayaLite(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        name: str = "AlayaLite",
        **kwargs,
    ) -> None:
        """Initialize wrapper around the vector database client.
        Args:
            dim(int): the dimension of the dataset
            db_config(dict): configs to establish connections with the vector database
            db_case_config(DBCaseConfig | None): case specific configs for indexing and searching
            drop_old(bool): whether to drop the existing collection of the dataset.
        """
        log.warning(f"Alayalite init, drop old: {drop_old}")
        self.name = name
        self._dim = dim
        self._db_config = db_config
        self._case_config = db_case_config
        self._collection_name = collection_name
        self._drop_old = drop_old
        self._client = alayalite.Client(**self._db_config)
        if drop_old:
            if self._collection_name in self._client.list_collections():
                self._client.delete_collection(collection_name=self._collection_name, delete_on_disk=True)

        self._client = None

    @contextmanager
    def init(self) -> None: # type: ignore
        """create and destory connections to database.
        Why contextmanager:

            In multiprocessing search tasks, vectordbbench might init
            totally hundreds of thousands of connections with DB server.

            Too many connections may drain local FDs or server connection resources.
            If the DB client doesn't have `close()` method, just set the object to None.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """

        # self._collection = self._client.get_or_create_collection(name=self._collection_name)
        # yield
        # self._client.save_collection(self._collection_name)
        # self._collection = None
        try:
            self._client = alayalite.Client(**self._db_config)
            self._collection = self._client.get_or_create_collection(self._collection_name  )
            yield
        finally:
            self._client = None
            self._collection = None
        
    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert the embeddings to the vector database. The default number of embeddings for
        each insert_embeddings is 5000.

        Args:
            embeddings(list[list[float]]): list of embedding to add to the vector database.
            metadatas(list[int]): metadata associated with the embeddings, for filtering.
            **kwargs(Any): vector database specific parameters.

        Returns:
            int: inserted data count
        """
        log.info(f"[alayalite] insert_embeddings {len(embeddings)}")
        assert self._collection is not None, "Please call self.init() before"
        assert len(embeddings) == len(metadata)
        size = len(embeddings)
        self._collection.insert(list(zip(metadata, metadata, embeddings, metadata)))
        self._client.save_collection(self._collection_name)
        log.warning(f"insert embedding: index py {self._collection._Collection__index_py}")
        log.info(f"[alayalite] insert_embeddings done")
        return size, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.

        Returns:
            list[int]: list of k most similar embeddings IDs to the query embedding.
        """
        assert self._collection is not None, "Please call self.init() before"
        res =  self._collection.batch_query(
            [query], k, **self._case_config.search_param()
        )
        return res["id"][0]


    def optimize(self, data_size: int | None = None):
        """optimize will be called between insertion and search in performance cases.

        Should be blocked until the vectorDB is ready to be tested on
        heavy performance cases.

        Time(insert the dataset) + Time(optimize) will be recorded as "load_duration" metric
        Optimize's execution time is limited, the limited time is based on cases.
        """
        log.warning(f"optimize: index py {self._collection._Collection__index_py}")
        return