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

import os
import uuid
import warnings
from typing import Any, Sequence

import pytest
import pytest_asyncio
from llama_index.core.schema import MetadataMode, NodeRelationship, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from llama_index_alloydb_pg import AlloyDBEngine, Column
from llama_index_alloydb_pg.async_vector_store import AsyncAlloyDBVectorStore
from llama_index_alloydb_pg.indexes import HNSWQueryOptions, ScaNNQueryOptions

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())
DEFAULT_TABLE_CUSTOM_VS = "test_table" + str(uuid.uuid4())
HYBRID_TABLE = "test_table" + str(uuid.uuid4())
VECTOR_SIZE = 768

texts = ["foo", "bar", "baz", "foobar"]
embedding = [1.0] * VECTOR_SIZE
nodes = [
    TextNode(
        id_=str(uuid.uuid4()),
        text=texts[i],
        embedding=[1 / (i + 1.0)] * VECTOR_SIZE,
    )
    for i in range(len(texts))
]
# setting each node as their own parent
for node in nodes:
    node.relationships[NodeRelationship.SOURCE] = node.as_related_node_info()
# Cosine distances to [1, 0, 0]: b 0, a 0.2, d 0.5, c 1.
# Text matches for "database": c (twice) ranks above a.
hybrid_nodes = [
    TextNode(id_=id_, text=text_, embedding=emb, metadata={"lang": lang})
    for id_, text_, emb, lang in [
        ("a", "postgres database", [0.8, 0.6, 0.0], "en"),
        ("b", "vector search engine", [1.0, 0.0, 0.0], "en"),
        ("c", "database database tuning", [0.0, 0.0, 1.0], "fr"),
        ("d", "cooking recipes", [0.5, 0.0, 0.866025], "en"),
    ]
]
sync_method_exception_str = "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(engine: AlloyDBEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


async def afetch(engine: AlloyDBEngine, query: str) -> Sequence[RowMapping]:
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        result_fetch = result_map.fetchall()
    return result_fetch


@pytest.mark.asyncio(loop_scope="class")
class TestVectorStore:
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
        return get_env_var("DB_USER", "database name on AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_pwd(self) -> str:
        return get_env_var("DB_PASSWORD", "database name on AlloyDB instance")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            cluster=db_cluster,
            region=db_region,
            database=db_name,
        )

        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_CUSTOM_VS}"')
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine._ainit_vector_store_table(
            DEFAULT_TABLE, VECTOR_SIZE, overwrite_existing=True
        )
        vs = await AsyncAlloyDBVectorStore.create(engine, table_name=DEFAULT_TABLE)
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def custom_vs(self, engine):
        await engine._ainit_vector_store_table(
            DEFAULT_TABLE_CUSTOM_VS,
            VECTOR_SIZE,
            overwrite_existing=True,
            metadata_columns=[
                Column(name="len", data_type="INTEGER", nullable=False),
                Column(
                    name="nullable_int_field",
                    data_type="INTEGER",
                    nullable=True,
                ),
                Column(
                    name="nullable_str_field",
                    data_type="VARCHAR",
                    nullable=True,
                ),
            ],
        )
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            table_name=DEFAULT_TABLE_CUSTOM_VS,
            metadata_columns=[
                "len",
                "nullable_int_field",
                "nullable_str_field",
            ],
            index_query_options=HNSWQueryOptions(ef_search=1),
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def custom_vs_scann(self, engine, custom_vs):
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            table_name=DEFAULT_TABLE_CUSTOM_VS,
            metadata_columns=[
                "len",
                "nullable_int_field",
                "nullable_str_field",
            ],
            index_query_options=ScaNNQueryOptions(
                num_leaves_to_search=1, pre_reordering_num_neighbors=2
            ),
        )
        yield vs

    async def test_init_with_constructor(self, engine):
        key = object()
        with pytest.raises(Exception):
            AsyncAlloyDBVectorStore(key, engine, table_name=DEFAULT_TABLE)

    async def test_validate_id_column_create(self, engine, vs):
        test_id_column = "test_id_column"
        with pytest.raises(
            Exception, match=f"Id column, {test_id_column}, does not exist."
        ):
            await AsyncAlloyDBVectorStore.create(
                engine, table_name=DEFAULT_TABLE, id_column=test_id_column
            )

    async def test_validate_text_column_create(self, engine, vs):
        test_text_column = "test_text_column"
        with pytest.raises(
            Exception, match=f"Text column, {test_text_column}, does not exist."
        ):
            await AsyncAlloyDBVectorStore.create(
                engine, table_name=DEFAULT_TABLE, text_column=test_text_column
            )

    async def test_validate_embedding_column_create(self, engine, vs):
        test_embed_column = "test_embed_column"
        with pytest.raises(
            Exception,
            match=f"Embedding column, {test_embed_column}, does not exist.",
        ):
            await AsyncAlloyDBVectorStore.create(
                engine,
                table_name=DEFAULT_TABLE,
                embedding_column=test_embed_column,
            )

    async def test_validate_node_column_create(self, engine, vs):
        test_node_column = "test_node_column"
        with pytest.raises(
            Exception, match=f"Node column, {test_node_column}, does not exist."
        ):
            await AsyncAlloyDBVectorStore.create(
                engine, table_name=DEFAULT_TABLE, node_column=test_node_column
            )

    async def test_validate_ref_doc_id_column_create(self, engine, vs):
        test_ref_doc_id_column = "test_ref_doc_id_column"
        with pytest.raises(
            Exception,
            match=f"Reference Document Id column, {test_ref_doc_id_column}, does not exist.",
        ):
            await AsyncAlloyDBVectorStore.create(
                engine,
                table_name=DEFAULT_TABLE,
                ref_doc_id_column=test_ref_doc_id_column,
            )

    async def test_validate_metadata_json_column_create(self, engine, vs):
        test_metadata_json_column = "test_metadata_json_column"
        with pytest.raises(
            Exception,
            match=f"Metadata column, {test_metadata_json_column}, does not exist.",
        ):
            await AsyncAlloyDBVectorStore.create(
                engine,
                table_name=DEFAULT_TABLE,
                metadata_json_column=test_metadata_json_column,
            )

    async def test_async_add(self, engine, vs):
        await vs.async_add(nodes)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 4

    async def test_async_add_custom_vs(self, engine, custom_vs):
        # setting extra metadata to be indexed in separate column
        for node in nodes:
            node.metadata["len"] = len(node.text)

        await custom_vs.async_add(nodes)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE_CUSTOM_VS}"')
        assert len(results) == 4
        assert results[0]["len"] == 3
        assert results[0]["nullable_int_field"] == None
        assert results[0]["nullable_str_field"] == None

    async def test_adelete(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await vs.async_add(nodes)
        await vs.adelete(nodes[0].node_id)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

    async def test_adelete_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await vs.async_add(nodes)
        await vs.adelete_nodes(
            node_ids=[nodes[0].node_id, nodes[1].node_id],
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text",
                        value="foo",
                        operator=FilterOperator.TEXT_MATCH,
                    ),
                    MetadataFilter(key="text", value="bar", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        )

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2

    async def test_aget_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await vs.async_add(nodes)
        results = await vs.aget_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text",
                        value="foo",
                        operator=FilterOperator.TEXT_MATCH,
                    ),
                    MetadataFilter(
                        key="text",
                        value="bar",
                        operator=FilterOperator.TEXT_MATCH,
                    ),
                ],
                condition=FilterCondition.AND,
            )
        )

        assert len(results) == 1
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobar"

    async def test_aquery(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await vs.async_add(nodes)
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, similarity_top_k=3
        )
        results = await vs.aquery(query)

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3
        assert results.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_aquery_scann(self, engine, custom_vs_scann):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE_CUSTOM_VS}"')
        # setting extra metadata to be indexed in separate column
        for node in nodes:
            node.metadata["len"] = len(node.text)

        await custom_vs_scann.async_add(nodes)
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, similarity_top_k=3
        )
        results = await custom_vs_scann.aquery(query)

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3

    async def test_aquery_filters(self, engine, custom_vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE_CUSTOM_VS}"')
        # setting extra metadata to be indexed in separate column
        for node in nodes:
            node.metadata["len"] = len(node.text)

        await custom_vs.async_add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="some_test_column",
                    value=["value_should_be_ignored"],
                    operator=FilterOperator.CONTAINS,
                ),
                MetadataFilter(
                    key="len",
                    value=3,
                    operator=FilterOperator.LTE,
                ),
                MetadataFilter(
                    key="len",
                    value=3,
                    operator=FilterOperator.GTE,
                ),
                MetadataFilter(
                    key="len",
                    value=2,
                    operator=FilterOperator.GT,
                ),
                MetadataFilter(
                    key="len",
                    value=4,
                    operator=FilterOperator.LT,
                ),
                MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="len",
                            value=6.0,
                            operator=FilterOperator.NE,
                        ),
                    ],
                    condition=FilterCondition.OR,
                ),
            ],
            condition=FilterCondition.AND,
        )
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, filters=filters, similarity_top_k=-1
        )
        with warnings.catch_warnings(record=True) as w:
            results = await custom_vs.aquery(query)

            assert len(w) == 1
            assert "Expecting a scalar in the filter value" in str(w[-1].message)

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3

    async def test_aclear(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_adelete
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await vs.async_add(nodes)
        await vs.aclear()

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 0

    async def test_add(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.add(nodes)

    async def test_get_nodes(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.get_nodes()

    async def test_query(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.query(VectorStoreQuery(query_str="foo"))

    async def test_delete(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.delete("test_ref_doc_id")

    async def test_delete_nodes(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.delete_nodes(["test_node_id"])

    async def test_clear(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.clear()

    async def test_parse_metadata_filters_parameterizes_key_and_value(self, custom_vs):
        injected_key = "k' IS NULL OR 1=1 OR li_metadata->>'k"
        injected_value = "x') OR 1=1;--"

        bind_params: dict[str, Any] = {}
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key=injected_key,
                    value=injected_value,
                    operator=FilterOperator.EQ,
                ),
                MetadataFilter(
                    key="len",
                    value=3,
                    operator=FilterOperator.EQ,
                ),
                MetadataFilter(
                    key="tags",
                    value='a"b',
                    operator=FilterOperator.CONTAINS,
                ),
            ],
            condition=FilterCondition.AND,
        )

        clause = custom_vs._AsyncAlloyDBVectorStore__parse_metadata_filters_recursively(
            filters, bind_params
        )

        assert injected_key not in clause
        assert injected_value not in clause
        assert (
            clause
            == 'li_metadata->>:key_0 = :param_1 and "len"::float = :param_2 and (li_metadata->>:key_3)::jsonb @> :param_4 '
        )
        assert bind_params == {
            "key_0": injected_key,
            "param_1": injected_value,
            "param_2": 3,
            "key_3": "tags",
            "param_4": '["a\\"b"]',
        }

        results = await custom_vs.aget_nodes(filters=filters)
        assert len(results) == 0

    @pytest_asyncio.fixture(scope="class")
    async def hybrid_vs(self, engine):
        await engine._ainit_vector_store_table(HYBRID_TABLE, 3, overwrite_existing=True)
        vs = await AsyncAlloyDBVectorStore.create(engine, table_name=HYBRID_TABLE)
        await vs.async_add(hybrid_nodes)
        yield vs
        await aexecute(engine, f'DROP TABLE IF EXISTS "{HYBRID_TABLE}"')

    async def test_aquery_default_unchanged(self, hybrid_vs):
        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0], query_str="database", similarity_top_k=4
        )
        results = await hybrid_vs.aquery(query)
        assert results.ids == ["b", "a", "d", "c"]
        assert results.similarities == pytest.approx([0.0, 0.2, 0.5, 1.0], abs=1e-5)

    @pytest.mark.parametrize(
        "mode", [VectorStoreQueryMode.TEXT_SEARCH, VectorStoreQueryMode.SPARSE]
    )
    async def test_aquery_text_search(self, hybrid_vs, mode):
        query = VectorStoreQuery(query_str="database", mode=mode, similarity_top_k=4)
        results = await hybrid_vs.aquery(query)
        assert results.ids == ["c", "a"]
        assert results.nodes[0].get_content() == "database database tuning"
        assert results.similarities[0] > results.similarities[1] > 0

        query.sparse_top_k = 1
        results = await hybrid_vs.aquery(query)
        assert results.ids == ["c"]

    async def test_aquery_text_search_with_tsvector_column(self, engine, hybrid_vs):
        await aexecute(
            engine,
            f"""ALTER TABLE "{HYBRID_TABLE}" ADD COLUMN IF NOT EXISTS tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', text)) STORED""",
        )
        vs = await AsyncAlloyDBVectorStore.create(
            engine,
            table_name=HYBRID_TABLE,
            hybrid_search_column="tsv",
            text_search_config="english",
        )
        query = VectorStoreQuery(
            query_str="databases",
            mode=VectorStoreQueryMode.TEXT_SEARCH,
            similarity_top_k=4,
        )
        results = await vs.aquery(query)
        assert results.ids == ["c", "a"]

        query.query_embedding = [1.0, 0.0, 0.0]
        query.mode = VectorStoreQueryMode.HYBRID
        results = await vs.aquery(query)
        assert results.ids == ["a", "c", "b", "d"]

    async def test_create_with_invalid_hybrid_search_column(self, engine, hybrid_vs):
        with pytest.raises(ValueError, match="Hybrid search column, missing, does not"):
            await AsyncAlloyDBVectorStore.create(
                engine, table_name=HYBRID_TABLE, hybrid_search_column="missing"
            )
        with pytest.raises(ValueError, match="is not type tsvector"):
            await AsyncAlloyDBVectorStore.create(
                engine, table_name=HYBRID_TABLE, hybrid_search_column="text"
            )

    async def test_aquery_hybrid(self, hybrid_vs):
        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0],
            query_str="database",
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=4,
        )
        results = await hybrid_vs.aquery(query)
        # a and c match both sides, b and d match the dense side only.
        assert results.ids == ["a", "c", "b", "d"]
        assert results.similarities == pytest.approx(
            [1 / 62, 0.5 / 64 + 0.5 / 61, 0.5 / 61, 0.5 / 63]
        )

        query.alpha = 1.0
        results = await hybrid_vs.aquery(query)
        assert results.ids == ["b", "a", "d", "c"]

        query.alpha = 0.0
        query.hybrid_top_k = 2
        results = await hybrid_vs.aquery(query)
        assert results.ids == ["c", "a"]

    async def test_aquery_hybrid_with_filters(self, hybrid_vs):
        query = VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0],
            query_str="database",
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=4,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="lang", value="en")],
            ),
        )
        results = await hybrid_vs.aquery(query)
        assert results.ids == ["a", "b", "d"]

        query.mode = VectorStoreQueryMode.TEXT_SEARCH
        results = await hybrid_vs.aquery(query)
        assert results.ids == ["a"]

    async def test_aquery_hybrid_invalid_input(self, hybrid_vs):
        with pytest.raises(ValueError, match="query_str is required"):
            await hybrid_vs.aquery(
                VectorStoreQuery(query_str=" ", mode=VectorStoreQueryMode.TEXT_SEARCH)
            )
        with pytest.raises(ValueError, match="query_embedding is required"):
            await hybrid_vs.aquery(
                VectorStoreQuery(query_str="database", mode=VectorStoreQueryMode.HYBRID)
            )
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            await hybrid_vs.aquery(
                VectorStoreQuery(
                    query_embedding=[1.0, 0.0, 0.0],
                    query_str="database",
                    mode=VectorStoreQueryMode.HYBRID,
                    alpha=1.5,
                )
            )
