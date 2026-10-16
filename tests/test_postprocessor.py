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
from typing import Any, List, Optional

import pytest
import pytest_asyncio
from llama_index.core import MockEmbedding, StorageContext, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, CBEventType, LlamaDebugHandler
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.no_text import NoText
from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle, TextNode
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBRerank, AlloyDBVectorStore
from llama_index_alloydb_pg.postprocessor import _to_ranked_nodes, dispatcher

RANKING_MODEL_ID = "semantic-ranker-default-003"
# google_ml.rank raises (SQLSTATE P0001) unless this database flag is on.
AI_QUERY_ENGINE_FLAG = "google_ml_integration.enable_ai_query_engine"
FLAG_QUERY = "SELECT current_setting(:flag, true)"
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


class RerankEvents(BaseEventHandler):
    events: List[Any] = []

    def handle(self, event: Any, **kwargs: Any) -> None:
        self.events.append(event)


def check_ranked(result: List[NodeWithScore], expected_len: int) -> None:
    assert len(result) == expected_len
    assert result[0].node is NODES[1].node
    scores = [node.score for node in result]
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

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            region=db_region,
            cluster=db_cluster,
            instance=db_instance,
            database=db_name,
        )
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def ranking(self, engine):
        async def fetch_flag() -> Optional[str]:
            async with engine._pool.connect() as conn:
                result = await conn.execute(
                    text(FLAG_QUERY), {"flag": AI_QUERY_ENGINE_FLAG}
                )
                return result.scalar()

        if await engine._run_as_async(fetch_flag()) != "on":
            pytest.skip(f"google_ml.rank needs the {AI_QUERY_ENGINE_FLAG} flag on.")

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
        assert AlloyDBRerank.class_name() == "AlloyDBRerank"

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

    async def test_nodes_without_text(self, engine, ranking):
        # Blank and image nodes are left out, and top_n may exceed the rest.
        # Format characters (ZWSP, BOM, ZWJ, word joiner) count as blank.
        empty = NodeWithScore(node=TextNode(text=" \n\t\u200b\ufeff\u200d\u2060"))
        image = NodeWithScore(node=ImageNode(image_url="https://example.com/a.png"))
        nodes = [empty, NODES[0], image, NODES[1], NODES[2]]
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID, top_n=10)
        expected = await reranker.apostprocess_nodes(NODES, query_str=QUERY)
        check_ranked(expected, expected_len=3)
        handler = RerankEvents()
        dispatcher.add_event_handler(handler)
        assert await reranker.apostprocess_nodes(nodes, query_str=QUERY) == expected
        dispatcher.event_handlers.remove(handler)
        assert [event.nodes for event in handler.events] == [NODES, expected]
        assert reranker.postprocess_nodes(nodes, query_str=QUERY) == expected
        for blank in ([], [empty, image]):
            assert await reranker.apostprocess_nodes(blank, query_str=QUERY) == []
            assert reranker.postprocess_nodes(blank, query_str=QUERY) == []
        assert [node.score for node in NODES] == [0.5, 0.5, 0.5]
        assert all(new is not old for new in expected for old in NODES)

    async def test_database_error(self, engine, ranking):
        # A NUL byte fails with SQLSTATE 22021, raised unchanged.
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID)
        nodes = [NodeWithScore(node=TextNode(text="abc\x00def"))]
        error = "CharacterNotInRepertoireError.*invalid byte sequence"
        with pytest.raises(DBAPIError, match=error):
            await reranker.apostprocess_nodes(nodes, query_str=QUERY)
        with pytest.raises(DBAPIError, match=error):
            reranker.postprocess_nodes(nodes, query_str=QUERY)

    async def test_missing_query(self, engine):
        reranker = AlloyDBRerank(engine=engine, model_id=RANKING_MODEL_ID)
        for query in (None, "", "  \n\t "):
            match = "Missing query bundle" if query is None else "Query string cannot"
            with pytest.raises(ValueError, match=match):
                await reranker.apostprocess_nodes(NODES, query_str=query)
            with pytest.raises(ValueError, match=match):
                reranker.postprocess_nodes(NODES, query_str=query)

    async def test_to_ranked_nodes(self):
        pairs = [(3, 0.4), (1, None), (0, 0.99), (4, 0.9), (2, 0.95)]
        rows = [{"index": index, "score": score} for index, score in pairs]
        result = _to_ranked_nodes(NODES, rows)  # type: ignore[arg-type]
        expected = [(NODES[1].node, 0.95), (NODES[2].node, 0.4)]
        assert [(node.node, node.score) for node in result] == expected
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
        for response in (await query_engine.aquery(QUERY), query_engine.query(QUERY)):
            [first, second] = response.source_nodes
            assert first.node.get_content() == TEXTS[1]
            assert first.score >= second.score  # type: ignore[operator]
