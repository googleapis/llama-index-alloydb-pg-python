# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding,
)
from llama_index.core.bridge.pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)
from sqlalchemy import text

from .engine import AlloyDBEngine

DEFAULT_NUM_WORKERS = 4

EMBEDDING_QUERY = """
    SELECT embedding(CAST(:model_id AS TEXT), u.t) AS embedding
    FROM unnest(CAST(:texts AS TEXT[])) WITH ORDINALITY AS u(t, ord)
    ORDER BY u.ord
"""


class AlloyDBEmbedding(BaseEmbedding):
    """Embedding model that generates embeddings inside AlloyDB.

    Texts are embedded with the AlloyDB AI ``embedding(model_id, content)``
    function, so ``model_id`` must be a model registered with model endpoint
    management, for example ``text-embedding-005``. Each batch of
    ``embed_batch_size`` texts is one SQL statement, and ``embedding()`` runs
    once per row, so a larger batch saves round trips but not model calls.
    An empty string raises ``ValueError``. The batch methods check their whole
    input, because LlamaIndex passes the hooks one batch at a time, so a hook
    could not name the text's position.

    The async batch methods embed ``num_workers`` batches at a time, each on its
    own pool connection. ``DEFAULT_NUM_WORKERS`` (4) fits the engine's default
    pool. More workers than the pool allows, or ``None``, 0 or 1 (no limit in
    llama-index-core), can fail with ``sqlalchemy.exc.TimeoutError``. The sync
    methods need an engine with a background event loop, such as one from
    ``AlloyDBEngine.from_instance`` or ``AlloyDBEngine.afrom_instance``.
    """

    model_config = ConfigDict(validate_assignment=True)
    model_id: str = Field(
        description="AlloyDB model endpoint id, e.g. 'text-embedding-005'."
    )
    _engine: AlloyDBEngine = PrivateAttr()

    def __init__(
        self,
        engine: AlloyDBEngine,
        model_id: str,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        num_workers: Optional[int] = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """AlloyDBEmbedding constructor.

        Args:
            engine (AlloyDBEngine): AlloyDB engine with a connection pool to the database.
            model_id (str): The model endpoint id used to generate embeddings.
            embed_batch_size (int): The number of texts sent in one SQL statement.
                Defaults to DEFAULT_EMBED_BATCH_SIZE.
            num_workers (Optional[int]): The number of batches the async batch
                methods embed at a time. Defaults to DEFAULT_NUM_WORKERS.
            **kwargs (Any): Passed to BaseEmbedding. model_name always follows model_id.

        Raises:
            ValueError: If model_id is empty or blank.
        """
        if not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        # model_id is a field of this class, unknown to BaseEmbedding.__init__
        super().__init__(  # type: ignore[call-arg]
            model_id=model_id,
            embed_batch_size=embed_batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self._engine = engine

    @model_validator(mode="after")
    def __sync_model_name(self) -> AlloyDBEmbedding:
        """Keep model_name equal to model_id, on construction and assignment."""
        if self.model_name != self.model_id:
            self.model_name = self.model_id
        return self

    @classmethod
    def class_name(cls) -> str:
        return "AlloyDBEmbedding"

    @staticmethod
    def __check_texts(texts: List[str]) -> None:
        """Raise ValueError naming the position of the first empty text."""
        if "" in texts:
            raise ValueError(f"Text at position {texts.index('')} is empty.")

    def get_text_embedding_batch(
        self, texts: List[str], *args: Any, **kwargs: Any
    ) -> List[Embedding]:
        """Embed texts in batches, raising ValueError for an empty text."""
        self.__check_texts(texts)
        return super().get_text_embedding_batch(texts, *args, **kwargs)

    async def aget_text_embedding_batch(
        self, texts: List[str], *args: Any, **kwargs: Any
    ) -> List[Embedding]:
        """Embed texts in batches, raising ValueError for an empty text."""
        self.__check_texts(texts)
        return await super().aget_text_embedding_batch(texts, *args, **kwargs)

    async def __aembed(self, texts: List[str]) -> List[Embedding]:
        """Embed a batch of texts with one SQL statement, in input order."""
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(
                text(EMBEDDING_QUERY), {"model_id": self.model_id, "texts": texts}
            )
            rows = result.mappings().fetchall()
        return [row["embedding"] for row in rows]

    async def __aembed_one(self, text: str) -> Embedding:
        """Embed one text, raising ValueError if it is empty."""
        if not text:
            raise ValueError("Cannot embed an empty string.")
        return (await self.__aembed([text]))[0]

    def _get_query_embedding(self, query: str) -> Embedding:
        """Embed a query synchronously."""
        return self._engine._run_as_sync(self.__aembed_one(query))

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Embed a query asynchronously."""
        return await self._engine._run_as_async(self.__aembed_one(query))

    def _get_text_embedding(self, text: str) -> Embedding:
        """Embed a text synchronously."""
        return self._engine._run_as_sync(self.__aembed_one(text))

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Embed a text asynchronously."""
        return await self._engine._run_as_async(self.__aembed_one(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Embed a batch of texts synchronously."""
        return self._engine._run_as_sync(self.__aembed(texts))

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Embed a batch of texts asynchronously."""
        return await self._engine._run_as_async(self.__aembed(texts))
