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
from llama_index.core.schema import TextNode  # type: ignore
from llama_index.core.vector_stores.types import VectorStoreQuery
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from llama_index_alloydb_pg import AlloyDBEngine
from llama_index_alloydb_pg.async_vectorstore import AsyncAlloyDBVectorStore

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())

texts = ["foo", "bar", "baz"]
nodes = [TextNode(text=texts[i]) for i in range(len(texts))]
sync_method_exception_str = "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


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
    async def engine(
        self, db_project, db_region, db_cluster, db_instance, db_name, db_user, db_pwd
    ):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            cluster=db_cluster,
            region=db_region,
            database=db_name,
            user=db_user,
            password=db_pwd,
        )

        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        vs = await AsyncAlloyDBVectorStore.create(
            engine, table_name=DEFAULT_TABLE, perform_validation=False
        )
        yield vs

    async def test_init_with_constructor(self, engine):
        with pytest.raises(Exception):
            AsyncAlloyDBVectorStore(engine, table_name=DEFAULT_TABLE)

    async def test_validate_columns_create(self, engine):
        # TODO: add tests for more columns after engine::init is implemented
        # currently, since there's no table first validation condition fails.
        test_id_column = "test_id_column"
        with pytest.raises(
            Exception, match=f"Id column, {test_id_column}, does not exist."
        ):
            await AsyncAlloyDBVectorStore.create(
                engine, table_name="non_existing_table", id_column=test_id_column
            )

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
