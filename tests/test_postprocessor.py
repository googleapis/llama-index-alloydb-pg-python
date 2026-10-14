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
from typing import List, Optional

import pytest
import pytest_asyncio
from llama_index.core import MockEmbedding, StorageContext, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, CBEventType, LlamaDebugHandler
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.no_text import NoText
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBRerank, AlloyDBVectorStore
from llama_index_alloydb_pg.postprocessor import _to_ranked_nodes

RANKING_MODEL_ID = "semantic-ranker-default-003"
# google_ml.rank raises (SQLSTATE P0001) unless this database flag is on.
AI_QUERY_ENGINE_FLAG = "google_ml_integration.enable_ai_query_engine"
QUERY = "neural networks"
TABLE_NAME = "test_table" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 8
TEXTS = [
    "Alpha: Introduction to Machine Learning.",
    "Beta: Deep Learning and Neural Networks.",
    "Gamma: Advanced Database Systems in Cloud.",
]
NODES = [
    NodeWithScore(node=TextNode(text=content, id_=str(uuid.uuid4())), score=0.5)
    for content in TEXTS
]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


def check_ranked(result: List[NodeWithScore], expected_len: int) -> None:
    assert len(result) == expected_len
    assert result[0].node is NODES[1].node
    scores = [node.score for node in result]
    assert all(isinstance(score, float) for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


@pytest.mark.asyncio(loop_scope="class")
class TestAlloyDBRerank:
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
    def db_password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for AlloyDB")

    @pytest_asyncio.fixture(scope="class")
    async def engine(
        self,
        db_project,
        db_region,
        db_cluster,
        db_instance,
        db_name,
        db_user,
        db_password,
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            region=db_region,
            cluster=db_cluster,
            instance=db_instance,
            database=db_name,
            user=db_user,
            password=db_password,
        )
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def ranking(self, engine):
        """Skips the test if the AI query engine flag is off on this instance."""
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID)
        try:
            await reranker.apostprocess_nodes(NODES[:1], query_str=QUERY)
        except DBAPIError as e:
            if getattr(e.orig, "sqlstate", None) == "P0001" and (
                await self.ai_query_engine_flag(engine) != "on"
            ):
                pytest.skip(
                    "google_ml.rank is disabled on this instance (SQLSTATE P0001): "
                    f"the {AI_QUERY_ENGINE_FLAG} flag is off."
                )
            raise

    async def ai_query_engine_flag(self, engine: AlloyDBEngine) -> Optional[str]:
        async def fetch() -> Optional[str]:
            async with engine._pool.connect() as conn:
                result = await conn.execute(
                    text("SELECT current_setting(:flag, true)"),
                    {"flag": AI_QUERY_ENGINE_FLAG},
                )
                return result.scalar()

        return await engine._run_as_async(fetch())

    @pytest_asyncio.fixture(scope="class")
    async def index(self, engine):
        await engine.ainit_vector_store_table(TABLE_NAME, VECTOR_SIZE)
        vector_store = await AlloyDBVectorStore.create(engine, table_name=TABLE_NAME)
        index = VectorStoreIndex(
            [node.node for node in NODES],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=MockEmbedding(embed_dim=VECTOR_SIZE),
        )
        yield index

        async def drop_table() -> None:
            async with engine._pool.connect() as conn:
                await conn.execute(text(f'DROP TABLE IF EXISTS "{TABLE_NAME}"'))
                await conn.commit()

        await engine._run_as_async(drop_table())

    async def test_validation(self, engine):
        with pytest.raises(ValueError, match="model_id must be a non-empty"):
            AlloyDBRerank(engine=engine, model_id="  ")
        for top_n in (0, -1):
            with pytest.raises(ValueError, match="top_n must be a positive integer"):
                AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID, top_n=top_n)
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID)
        assert reranker.class_name() == "AlloyDBRerank"
        assert (reranker.model_id, reranker.top_n) == (RANKING_MODEL_ID, 2)

    async def test_apostprocess_nodes(self, engine, ranking):
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID)
        result = await reranker.apostprocess_nodes(NODES, query_str=QUERY)
        check_ranked(result, expected_len=2)

    async def test_postprocess_nodes(self, engine, ranking):
        handler = LlamaDebugHandler()
        reranker = AlloyDBRerank(
            engine=engine,
            model_id=RANKING_MODEL_ID,
            top_n=1,
            callback_manager=CallbackManager([handler]),
        )
        result = reranker.postprocess_nodes(NODES, QueryBundle(QUERY))
        check_ranked(result, expected_len=1)
        [(start, end)] = handler.get_event_pairs(CBEventType.RERANKING)
        assert start.payload["query_str"] == QUERY
        assert end.payload["nodes"] == result

    async def test_top_n_larger_than_nodes(self, engine, ranking):
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID, top_n=10)
        result = await reranker.apostprocess_nodes(NODES, query_str=QUERY)
        check_ranked(result, expected_len=3)
        assert {id(node.node) for node in result} == {id(node.node) for node in NODES}

    async def test_inputs_not_modified(self, engine, ranking):
        nodes = [NodeWithScore(node=node.node, score=0.5) for node in NODES]
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID, top_n=3)
        result = await reranker.apostprocess_nodes(nodes, query_str=QUERY)
        assert [node.score for node in nodes] == [0.5, 0.5, 0.5]
        assert [node.node for node in nodes] == [node.node for node in NODES]
        assert all(new is not old for new in result for old in nodes)

    async def test_empty_nodes(self, engine):
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID)
        assert await reranker.apostprocess_nodes([], query_str=QUERY) == []
        assert reranker.postprocess_nodes([], query_str=QUERY) == []

    async def test_missing_query(self, engine):
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID)
        with pytest.raises(ValueError, match="Missing query bundle"):
            await reranker.apostprocess_nodes(NODES)
        with pytest.raises(ValueError, match="Missing query bundle"):
            reranker.postprocess_nodes(NODES)
        for query in ("", "  \n\t "):
            with pytest.raises(ValueError, match="Query string cannot be empty"):
                await reranker.apostprocess_nodes(NODES, query_str=query)
            with pytest.raises(ValueError, match="Query string cannot be empty"):
                reranker.postprocess_nodes(NODES, query_str=query)

    async def test_to_ranked_nodes(self):
        rows = [
            {"index": 3, "score": 0.4},
            {"index": 1, "score": None},
            {"index": 0, "score": 0.99},
            {"index": 4, "score": 0.9},
            {"index": 2, "score": 0.95},
        ]
        result = _to_ranked_nodes(NODES, rows)  # type: ignore[arg-type]
        assert [(node.node, node.score) for node in result] == [
            (NODES[1].node, 0.95),
            (NODES[2].node, 0.4),
        ]
        assert _to_ranked_nodes(NODES, []) == []

    async def test_query_engine_end_to_end(self, engine, ranking, index):
        # MockEmbedding gives every text the same vector, so the retriever
        # order carries no relevance signal and the order comes from the
        # reranker. The NoText synthesizer never calls the LLM.
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID, top_n=2)
        query_engine = RetrieverQueryEngine(
            retriever=index.as_retriever(similarity_top_k=3),
            response_synthesizer=NoText(llm=MockLLM()),
            node_postprocessors=[reranker],
        )
        for response in (
            await query_engine.aquery(QUERY),
            query_engine.query(QUERY),
        ):
            nodes = response.source_nodes
            assert len(nodes) == 2
            assert nodes[0].node.get_content() == TEXTS[1]
            assert nodes[0].score is not None and nodes[1].score is not None
            assert nodes[0].score >= nodes[1].score
