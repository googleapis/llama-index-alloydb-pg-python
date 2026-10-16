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

import unicodedata
from typing import Any, List, Optional, Sequence, Tuple

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base import EventContext
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from sqlalchemy import RowMapping, text

from .engine import AlloyDBEngine

dispatcher = get_dispatcher(__name__)

# google_ml.rank works from extension 1.5.2 and is what ai.rank calls underneath.
# Casts are needed: it is overloaded and untyped binds are ambiguous (SQLSTATE 42725).
RANK_QUERY = """
    SELECT index, score FROM google_ml.rank(CAST(:model_id AS VARCHAR),
        CAST(:query AS TEXT), CAST(:documents AS TEXT[]), CAST(:top_n AS INTEGER))
"""


def _to_ranked_nodes(
    nodes: Sequence[NodeWithScore], rows: Sequence[RowMapping]
) -> List[NodeWithScore]:
    """Map 1-based ``(index, score)`` rows to new nodes, highest score first."""
    valid = [r for r in rows if r["score"] is not None and 0 < r["index"] <= len(nodes)]
    return [
        NodeWithScore(node=nodes[r["index"] - 1].node, score=float(r["score"]))
        for r in sorted(valid, key=lambda r: r["score"], reverse=True)
    ]


def _has_text(node: NodeWithScore) -> bool:
    """Whether the text has a character other than whitespace or format (Cf)."""
    content = node.node.get_content(MetadataMode.EMBED)
    return any(not c.isspace() and unicodedata.category(c) != "Cf" for c in content)


class AlloyDBRerank(BaseNodePostprocessor):
    """Reranks nodes with AlloyDB AI's ``google_ml.rank()`` function.

    The ranking runs inside the database through AlloyDB's Vertex AI
    integration. As listed in
    https://cloud.google.com/alloydb/docs/ai/rank-rerank-search-results-rag,
    it needs the ``google_ml_integration`` extension 1.5.2 or later, the
    ``google_ml_integration.enable_model_support`` and
    ``google_ml_integration.enable_ai_query_engine`` flags on, the Discovery
    Engine API enabled and ``roles/discoveryengine.viewer`` for the AlloyDB
    service agent. Nodes with no text (an ``ImageNode``) or only whitespace or
    format characters are left out. Database errors are raised unchanged.
    Async ``apostprocess_nodes`` needs llama-index-core 0.12.32 or later.
    """

    model_id: str = Field(
        description="AlloyDB ranking model id, e.g. 'semantic-ranker-default-003'."
    )
    top_n: int = Field(default=2, description="Number of nodes to return.")
    _engine: AlloyDBEngine = PrivateAttr()

    def __init__(
        self, engine: AlloyDBEngine, model_id: str, top_n: int = 2, **kwargs: Any
    ) -> None:
        """Create an AlloyDBRerank instance.

        Args:
            engine (AlloyDBEngine): The AlloyDB engine to run the ranking on.
            model_id (str): A Vertex AI ranking model id, for example
                ``"semantic-ranker-default-003"``, or a registered reranking
                model. Unregistered ids raise no error and rank with the
                default model, while a registered non-reranking model raises.
            top_n (int): Number of nodes to return. Defaults to 2. Checked
                only here: assigning it later is not validated.
            **kwargs (Any): Other ``BaseNodePostprocessor`` fields, for
                example ``callback_manager``.

        Raises:
            ValueError: If ``model_id`` is blank or ``top_n`` is less than 1.
        """
        if not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        if top_n < 1:
            raise ValueError("top_n must be a positive integer greater than 0.")
        # mypy only sees the fields of the base class in its __init__.
        super().__init__(model_id=model_id, top_n=top_n, **kwargs)  # type: ignore[call-arg]
        self._engine = engine

    @classmethod
    def class_name(cls) -> str:
        return "AlloyDBRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Rerank nodes by relevance to the query with google_ml.rank.

        Returns new ``NodeWithScore`` objects with the rank score, sorted by
        score (highest first) and limited to ``top_n``.

        Raises:
            ValueError: If ``query_bundle`` is missing or its query is blank.
        """
        query, nodes = self.__start(nodes, query_bundle)
        if not nodes:
            return []
        with self.callback_manager.event(
            CBEventType.RERANKING, payload=self.__payload(nodes, query)
        ) as event:
            rows = self._engine._run_as_sync(self.__arank(nodes, query))
            return self.__finish(event, nodes, rows)

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Async version of ``_postprocess_nodes``, run on the engine's loop."""
        query, nodes = self.__start(nodes, query_bundle)
        if not nodes:
            return []
        with self.callback_manager.event(
            CBEventType.RERANKING, payload=self.__payload(nodes, query)
        ) as event:
            rows = await self._engine._run_as_async(self.__arank(nodes, query))
            return self.__finish(event, nodes, rows)

    def __start(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> Tuple[str, List[NodeWithScore]]:
        """Skip nodes without text, emit the start event, validate the query."""
        # google_ml.rank fails on empty documents and ranks blank ones, so skip them.
        nodes = [n for n in nodes if _has_text(n)]
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model_id,
            )
        )
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if not query_bundle.query_str.strip():
            raise ValueError("Query string cannot be empty or whitespace.")
        return query_bundle.query_str, nodes

    def __payload(self, nodes: List[NodeWithScore], query: str) -> dict[str, Any]:
        """Build the payload of the reranking callback event."""
        return {
            EventPayload.NODES: nodes,
            EventPayload.MODEL_NAME: self.model_id,
            EventPayload.QUERY_STR: query,
            EventPayload.TOP_K: self.top_n,
        }

    def __finish(
        self,
        event: EventContext,
        nodes: List[NodeWithScore],
        rows: Sequence[RowMapping],
    ) -> List[NodeWithScore]:
        """Map the rows to new nodes and emit the rerank end events."""
        new_nodes = _to_ranked_nodes(nodes, rows)
        event.on_end(payload={EventPayload.NODES: new_nodes})
        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes

    async def __arank(
        self, nodes: List[NodeWithScore], query: str
    ) -> Sequence[RowMapping]:
        """Call google_ml.rank on the node texts."""
        documents = [n.node.get_content(MetadataMode.EMBED) for n in nodes]
        params = {
            "model_id": self.model_id,
            "query": query,
            "documents": documents,
            "top_n": self.top_n,
        }
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(RANK_QUERY), params)
            return result.mappings().fetchall()
