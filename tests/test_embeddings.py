# Copyright 2025 Google LLC
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

import os
import uuid
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from llama_index_alloydb_pg import AlloyDBEmbedding, AlloyDBEngine, AlloyDBModelManager


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


class TestAlloyDBEmbeddingUnit:
    def test_embed_query_inline(self):
        engine_mock = MagicMock()
        emb = AlloyDBEmbedding(engine=engine_mock, model_id="text-embedding-004")
        assert emb.embed_query_inline("sample query") == "embedding('text-embedding-004', 'sample query')::vector"

    def test_model_id_property(self):
        engine_mock = MagicMock()
        emb = AlloyDBEmbedding(engine=engine_mock, model_id="my-custom-model")
        assert emb.model_id == "my-custom-model"


@pytest.mark.asyncio
class TestAlloyDBEmbeddingIntegration:
    @pytest_asyncio.fixture
    async def engine(self):
        if not os.environ.get("PROJECT_ID"):
            pytest.skip("Skipping live cluster test: PROJECT_ID not set.")
        AlloyDBEngine._connector = None
        engine = await AlloyDBEngine.afrom_instance(
            project_id=get_env_var("PROJECT_ID", "GCP Project ID"),
            cluster=get_env_var("CLUSTER_ID", "AlloyDB Cluster ID"),
            instance=get_env_var("INSTANCE_ID", "AlloyDB Instance ID"),
            region=get_env_var("REGION", "AlloyDB Region"),
            database=get_env_var("DATABASE_ID", "Database Name"),
        )
        yield engine
        await engine.close()

    @pytest_asyncio.fixture
    async def embeddings(self, engine):
        model_id = "text_embedding_004_" + str(uuid.uuid4()).replace("-", "_")[:8]
        model_manager = await AlloyDBModelManager.create(engine=engine)
        await model_manager.acreate_model(
            model_id=model_id,
            model_provider="google",
            model_qualified_name="text-embedding-004",
            model_type="text_embedding",
        )
        emb = await AlloyDBEmbedding.acreate(engine=engine, model_id=model_id)
        yield emb
        await model_manager.adrop_model(model_id=model_id)

    async def test_embed_query(self, embeddings):
        embedding = embeddings.get_query_embedding("AlloyDB PostgreSQL vector search")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        for val in embedding:
            assert isinstance(val, float)

    async def test_aembed_query(self, embeddings):
        embedding = await embeddings.aget_query_embedding("AlloyDB PostgreSQL vector search")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        for val in embedding:
            assert isinstance(val, float)
