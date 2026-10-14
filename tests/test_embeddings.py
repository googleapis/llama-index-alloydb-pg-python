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

import os
import uuid
from typing import Sequence

import pytest
import pytest_asyncio
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from sqlalchemy import RowMapping, text
from sqlalchemy.exc import DBAPIError

from llama_index_alloydb_pg import AlloyDBEmbedding, AlloyDBEngine, AlloyDBVectorStore

MODEL_ID = "text-embedding-005"
VECTOR_SIZE = 768
SQL_TABLE = "test_embeddings_sql_" + str(uuid.uuid4()).replace("-", "_")
VS_TABLE = "test_embeddings_vs_" + str(uuid.uuid4()).replace("-", "_")

texts = [
    "The cat sat on the mat.",
    "AlloyDB is a PostgreSQL-compatible database service.",
    "Paris is the capital of France.",
    "Photosynthesis turns sunlight into chemical energy.",
    "The stock market closed higher today.",
]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(engine: AlloyDBEngine, query: str) -> None:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


async def afetch(engine: AlloyDBEngine, query: str) -> Sequence[RowMapping]:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    return await engine._run_as_async(run(engine, query))


@pytest.mark.asyncio(loop_scope="class")
class TestAlloyDBEmbeddingAsync:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for AlloyDB")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name on AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_user(self) -> str:
        return get_env_var("DB_USER", "database user for AlloyDB")

    @pytest.fixture(scope="module")
    def db_pwd(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for AlloyDB")

    @pytest_asyncio.fixture(scope="class")
    async def engine(
        self, db_project, db_region, db_cluster, db_instance, db_name, db_user, db_pwd
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            region=db_region,
            cluster=db_cluster,
            instance=db_instance,
            database=db_name,
            user=db_user,
            password=db_pwd,
        )
        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{SQL_TABLE}"')
        await engine.close()

    @pytest.fixture(scope="class")
    def embed_model(self, engine):
        return AlloyDBEmbedding(engine=engine, model_id=MODEL_ID)

    async def test_blank_model_id(self, engine):
        for model_id in ["", "   "]:
            with pytest.raises(ValueError, match="model_id"):
                AlloyDBEmbedding(engine=engine, model_id=model_id)

    async def test_aget_query_embedding(self, embed_model):
        embedding = await embed_model.aget_query_embedding("What is AlloyDB?")
        assert len(embedding) == VECTOR_SIZE
        assert all(isinstance(value, float) for value in embedding)
        assert embed_model.model_name == MODEL_ID
        assert embed_model.class_name() == "AlloyDBEmbedding"

    async def test_aget_text_embedding_batch(self, engine):
        # A batch size of 2 sends the 5 texts in 3 round trips.
        embed_model = AlloyDBEmbedding(
            engine=engine, model_id=MODEL_ID, embed_batch_size=2
        )
        batch = await embed_model.aget_text_embedding_batch(texts)
        singles = [await embed_model.aget_text_embedding(t) for t in texts]
        assert len(batch) == len(texts)
        for i, vector in enumerate(batch):
            scores = [embed_model.similarity(vector, single) for single in singles]
            assert scores[i] >= 0.999
            assert max(range(len(texts)), key=scores.__getitem__) == i

    async def test_embed_text_with_sql(self, engine, embed_model):
        await aexecute(engine, f'CREATE TABLE "{SQL_TABLE}" (id INT)')
        await aexecute(engine, f'INSERT INTO "{SQL_TABLE}" VALUES (1)')
        content = f'\'); DROP TABLE "{SQL_TABLE}"; --'
        embedding = await embed_model.aget_text_embedding(content)
        assert len(embedding) == VECTOR_SIZE
        rows = await afetch(engine, f'SELECT COUNT(*) AS n FROM "{SQL_TABLE}"')
        assert rows[0]["n"] == 1

    async def test_unknown_model(self, engine):
        embed_model = AlloyDBEmbedding(
            engine=engine, model_id="unknown-model-" + str(uuid.uuid4())
        )
        with pytest.raises(DBAPIError):
            await embed_model.aget_query_embedding("What is AlloyDB?")


@pytest.mark.asyncio(loop_scope="class")
class TestAlloyDBEmbeddingSync:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for AlloyDB")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name on AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_user(self) -> str:
        return get_env_var("DB_USER", "database user for AlloyDB")

    @pytest.fixture(scope="module")
    def db_pwd(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for AlloyDB")

    @pytest_asyncio.fixture(scope="class")
    async def engine(
        self, db_project, db_region, db_cluster, db_instance, db_name, db_user, db_pwd
    ):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            region=db_region,
            cluster=db_cluster,
            instance=db_instance,
            database=db_name,
            user=db_user,
            password=db_pwd,
        )
        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{VS_TABLE}"')
        await engine.close()

    @pytest.fixture(scope="class")
    def embed_model(self, engine):
        return AlloyDBEmbedding(engine=engine, model_id=MODEL_ID)

    async def test_get_query_embedding(self, embed_model):
        embedding = embed_model.get_query_embedding("What is AlloyDB?")
        assert len(embedding) == VECTOR_SIZE
        assert all(isinstance(value, float) for value in embedding)

    async def test_get_text_embedding_batch(self, embed_model):
        batch = embed_model.get_text_embedding_batch(texts)
        single = embed_model.get_text_embedding(texts[2])
        assert len(batch) == len(texts)
        assert embed_model.similarity(batch[2], single) >= 0.999

    async def test_vector_store_index_end_to_end(self, engine, embed_model):
        engine.init_vector_store_table(VS_TABLE, VECTOR_SIZE)
        vs = AlloyDBVectorStore.create_sync(engine, table_name=VS_TABLE)
        index = VectorStoreIndex(
            [TextNode(text=t) for t in texts],
            storage_context=StorageContext.from_defaults(vector_store=vs),
            embed_model=embed_model,
        )
        results = index.as_retriever(similarity_top_k=2).retrieve(
            "Which city is the capital of France?"
        )
        assert results[0].node.get_content() == texts[2]
        rows = await afetch(engine, f'SELECT COUNT(*) AS n FROM "{VS_TABLE}"')
        assert rows[0]["n"] == len(texts)
