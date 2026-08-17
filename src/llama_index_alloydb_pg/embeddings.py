# Copyright 2024 Google LLC
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

import json
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from sqlalchemy import text

from .engine import AlloyDBEngine
from .model_manager import AlloyDBModelManager


class AlloyDBEmbedding(BaseEmbedding):
    """Google AlloyDB Embeddings available via Model Endpoint Management."""

    model_id: str
    _engine: AlloyDBEngine

    def __init__(
        self,
        engine: AlloyDBEngine,
        model_id: str,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ):
        super().__init__(
            model_id=model_id,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._engine = engine
        self.model_id = model_id

    @classmethod
    async def acreate(
        cls, engine: AlloyDBEngine, model_id: str, **kwargs: Any
    ) -> AlloyDBEmbedding:
        """Create AlloyDBEmbedding instance asynchronously."""
        embeddings = cls(engine, model_id, **kwargs)
        model_exists = await embeddings.amodel_exists()
        if not model_exists:
            raise ValueError(f"Model {model_id} does not exist.")
        return embeddings

    @classmethod
    def create(
        cls, engine: AlloyDBEngine, model_id: str, **kwargs: Any
    ) -> AlloyDBEmbedding:
        """Create AlloyDBEmbedding instance synchronously."""
        embeddings = cls(engine, model_id, **kwargs)
        if not embeddings.model_exists():
            raise ValueError(f"Model {model_id} does not exist.")
        return embeddings

    async def amodel_exists(self) -> bool:
        """Checks if the embedding model exists asynchronously."""
        return await self._engine._run_as_async(self.__amodel_exists())

    def model_exists(self) -> bool:
        """Checks if the embedding model exists synchronously."""
        return self._engine._run_as_sync(self.__amodel_exists())

    async def __amodel_exists(self) -> bool:
        model_manager = await AlloyDBModelManager.create(self._engine)
        model = await model_manager.aget_model(model_id=self.model_id)
        return model is not None

    def embed_query_inline(self, query: str) -> str:
        """Return SQL expression for inline in-database embedding generation."""
        return f"embedding('{self.model_id}', '{query}')::vector"

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._engine._run_as_sync(self._aget_query_embedding(query))

    async def _aget_query_embedding(self, query: str) -> Embedding:
        sql = f"SELECT embedding('{self.model_id}', '{query}')::vector AS embedding"
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(sql))
            result_map = result.mappings()
            results = result_map.fetchall()
        val = results[0]["embedding"]
        if isinstance(val, str):
            return json.loads(val)
        return list(val)

    def _get_text_embedding(self, text_str: str) -> Embedding:
        return self._get_query_embedding(text_str)

    async def _aget_text_embedding(self, text_str: str) -> Embedding:
        return await self._aget_query_embedding(text_str)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return [await self._aget_text_embedding(t) for t in texts]
