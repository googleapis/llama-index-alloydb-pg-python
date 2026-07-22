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

from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBModel, AlloyDBModelManager


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


EMBEDDING_MODEL_NAME = "text_embedding_005_" + str(uuid.uuid4()).replace("-", "_")[:8]


class TestAlloyDBModelManagerUnit:
    def test_model_init(self):
        model = AlloyDBModel(
            model_id="test_model",
            model_request_url="http://test.url",
            model_provider="google",
            model_type="text_embedding",
            model_qualified_name="text-embedding-004",
            model_auth_type="user_credentials",
            model_auth_id="secret_id",
            input_transform_fn="input_fn",
            output_transform_fn="output_fn",
        )
        assert model.model_id == "test_model"
        assert model.model_provider == "google"
        assert model.model_qualified_name == "text-embedding-004"

    def test_direct_init_raises(self):
        engine_mock = MagicMock()
        with pytest.raises(Exception, match="Only create class through"):
            AlloyDBModelManager(object(), engine_mock)


@pytest.mark.asyncio
class TestAlloyDBModelManagerIntegration:
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

    @pytest_asyncio.fixture(scope="module")
    async def engine(self, db_project, db_region, db_cluster, db_instance, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            cluster=db_cluster,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="module")
    async def model_manager(self, engine):
        model_manager = await AlloyDBModelManager.create(engine)
        yield model_manager

    async def test_acreate_model(self, model_manager):
        if not os.environ.get("PROJECT_ID"):
            pytest.skip("Skipping live cluster test: PROJECT_ID not set.")
        await model_manager.acreate_model(
            model_id=EMBEDDING_MODEL_NAME,
            model_provider="google",
            model_qualified_name="text-embedding-004",
            model_type="text_embedding",
        )

    async def test_aget_model(self, model_manager):
        if not os.environ.get("PROJECT_ID"):
            pytest.skip("Skipping live cluster test: PROJECT_ID not set.")
        model_info = await model_manager.aget_model(model_id=EMBEDDING_MODEL_NAME)
        assert model_info is not None
        assert model_info.model_id == EMBEDDING_MODEL_NAME

    async def test_non_existent_model(self, model_manager):
        if not os.environ.get("PROJECT_ID"):
            pytest.skip("Skipping live cluster test: PROJECT_ID not set.")
        model_info = await model_manager.aget_model(model_id="non_existent_model_id")
        assert model_info is None

    async def test_adrop_model(self, model_manager):
        if not os.environ.get("PROJECT_ID"):
            pytest.skip("Skipping live cluster test: PROJECT_ID not set.")
        await model_manager.adrop_model(model_id=EMBEDDING_MODEL_NAME)
