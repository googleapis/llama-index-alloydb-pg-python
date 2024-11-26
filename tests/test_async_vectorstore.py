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
from typing import List, Sequence

import pytest
import pytest_asyncio
from llama_index.core.schema import NodeRelationship, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from llama_index_alloydb_pg import AlloyDBEngine
from llama_index_alloydb_pg.async_vectorstore import AsyncAlloyDBVectorStore

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())
VECTOR_SIZE = 768

texts = ["foo", "bar", "baz"]
embedding = [1.0] * VECTOR_SIZE
nodes = [
    TextNode(id_=str(uuid.uuid4()), text=texts[i], embedding=embedding)
    for i in range(len(texts))
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
        await aexecute(engine, f'DROP TABLE "{DEFAULT_TABLE}"')
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine._ainit_vector_store_table(
            DEFAULT_TABLE, VECTOR_SIZE, overwrite_existing=True
        )
        vs = await AsyncAlloyDBVectorStore.create(engine, table_name=DEFAULT_TABLE)
        yield vs

    async def test_init_with_constructor(self, engine):
        with pytest.raises(Exception):
            AsyncAlloyDBVectorStore(engine, table_name=DEFAULT_TABLE)

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
            Exception, match=f"Embedding column, {test_embed_column}, does not exist."
        ):
            await AsyncAlloyDBVectorStore.create(
                engine, table_name=DEFAULT_TABLE, embedding_column=test_embed_column
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
        # setting each node as their own parent
        for node in nodes:
            node.relationships[NodeRelationship.SOURCE] = node.as_related_node_info()

        await vs.async_add(nodes)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

    @pytest.mark.depends(on=["test_async_add"])
    async def test_adelete(self, engine, vs):
        await vs.adelete(nodes[0].node_id)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2

    @pytest.mark.depends(on=["test_adelete"])
    async def test_aclear(self, engine, vs):
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
