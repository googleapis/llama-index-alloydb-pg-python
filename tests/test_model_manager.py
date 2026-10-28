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
from dataclasses import asdict, fields

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBModel, AlloyDBModelManager
from llama_index_alloydb_pg.model_manager import MODEL_SUPPORT_FLAG

# google_ml_integration has no header function for Google endpoints (they
# authenticate with alloydb_service_agent_iam), so the create/get test uses the
# Anthropic one. The server only checks that the function exists.
HEADERS_FN = "google_ml.anthropic_claude_header_gen_fn"
QUALIFIED_NAME = "text-embedding-005"
REQUEST_URL = (
    "https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}"
    "/locations/{region}/publishers/google/models/text-embedding-005:predict"
)
EMBEDDING_MODEL = {
    "model_provider": "google",
    "model_type": "text_embedding",
    "model_qualified_name": QUALIFIED_NAME,
}


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


def new_model_id() -> str:
    return "li_test_model_" + uuid.uuid4().hex


async def askip_unless_model_support(engine: AlloyDBEngine) -> None:
    async def run():
        async with engine._pool.connect() as conn:
            query = text(
                "SELECT current_setting(:flag, true) AS flag, EXISTS (SELECT 1 FROM "
                "pg_extension WHERE extname = 'google_ml_integration') AS installed"
            )
            result = await conn.execute(query, {"flag": MODEL_SUPPORT_FLAG})
            return result.mappings().one()

    row = await engine._run_as_async(run())
    if not row["installed"] or row["flag"] != "on":
        pytest.skip(f"needs google_ml_integration and {MODEL_SUPPORT_FLAG}=on")


@pytest.fixture(scope="module")
def engine_args() -> dict:
    return dict(
        project_id=get_env_var("PROJECT_ID", "project id for google cloud"),
        region=get_env_var("REGION", "region for AlloyDB instance"),
        cluster=get_env_var("CLUSTER_ID", "cluster for AlloyDB"),
        instance=get_env_var("INSTANCE_ID", "instance for AlloyDB"),
        database=get_env_var("DATABASE_ID", "database name on AlloyDB instance"),
        user=get_env_var("DB_USER", "database user for AlloyDB"),
        password=get_env_var("DB_PASSWORD", "database password for AlloyDB"),
    )


@pytest.mark.asyncio(loop_scope="class")
class TestAlloyDBModelManagerAsync:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self, engine_args):
        engine = await AlloyDBEngine.afrom_instance(**engine_args)
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def manager(self, engine):
        await askip_unless_model_support(engine)
        yield await AlloyDBModelManager.create(engine)

    async def test_constructor_guard(self, engine):
        with pytest.raises(Exception, match="Only create class through"):
            AlloyDBModelManager(object(), engine)

    async def test_acreate_get_drop_model(self, manager, engine_args):
        model_id = new_model_id()
        url = REQUEST_URL.format(**engine_args)
        auth = "alloydb_service_agent_iam"
        expected = dict(EMBEDDING_MODEL, model_request_url=url, model_auth_type=auth)
        expected["generate_headers_fn"] = HEADERS_FN
        try:
            await manager.acreate_model(model_id, **expected)
            model = await manager.aget_model(model_id)
            assert asdict(model).items() >= dict(expected, model_id=model_id).items()
            await manager.adrop_model(model_id)
            assert await manager.aget_model(model_id) is None
        finally:
            await manager.adrop_model(model_id)

    async def test_alist_models(self, manager):
        model_id = new_model_id()
        try:
            await manager.acreate_model(model_id, **EMBEDDING_MODEL)
            models = {m.model_id: m for m in await manager.alist_models()}
            assert models[QUALIFIED_NAME].model_type == "text_embedding"
            assert models[model_id].model_qualified_name == QUALIFIED_NAME
        finally:
            await manager.adrop_model(model_id)

    async def test_aget_unknown_model(self, manager):
        assert await manager.aget_model(new_model_id()) is None

    async def test_model_id_with_quote(self, manager):
        model_id = "li_test_o'quote_" + uuid.uuid4().hex
        try:
            await manager.acreate_model(model_id, **EMBEDDING_MODEL)
            assert (await manager.aget_model(model_id)).model_id == model_id
            await manager.adrop_model(model_id)
            assert await manager.aget_model(model_id) is None
        finally:
            await manager.adrop_model(model_id)

    async def test_acreate_model_duplicate_id(self, manager):
        model_id = new_model_id()
        try:
            await manager.acreate_model(model_id, **EMBEDDING_MODEL)
            with pytest.raises(DBAPIError, match="Model already exists") as e:
                await manager.acreate_model(model_id, **EMBEDDING_MODEL)
            assert e.value.orig.sqlstate == "P0001"
        finally:
            await manager.adrop_model(model_id)

    @pytest.mark.parametrize("value", ["", "   ", 5])
    @pytest.mark.parametrize("name", [f.name for f in fields(AlloyDBModel)])
    async def test_acreate_model_invalid_argument(self, manager, name, value):
        model_id = new_model_id()
        args = dict(EMBEDDING_MODEL, model_id=model_id)
        args[name] = value
        with pytest.raises(ValueError, match=f"{name} must be a non-blank string"):
            await manager.acreate_model(**args)
        assert await manager.aget_model(model_id) is None

    async def test_model_round_trip(self, manager):
        model_id, copy_id = new_model_id(), new_model_id()
        try:
            await manager.acreate_model(model_id, **EMBEDDING_MODEL)
            args = asdict(await manager.aget_model(model_id))
            del args["model_id"]
            # The server fills in the transform functions for this model.
            transform = "google_ml.vertexai_text_embedding_{}_transform"
            assert args["model_in_transform_fn"] == transform.format("input")
            assert args["model_out_transform_fn"] == transform.format("output")
            await manager.acreate_model(copy_id, **args)
            copy = asdict(await manager.aget_model(copy_id))
            assert copy == dict(args, model_id=copy_id)
        finally:
            await manager.adrop_model(model_id)
            await manager.adrop_model(copy_id)


@pytest.mark.asyncio(loop_scope="class")
class TestAlloyDBModelManagerSync:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self, engine_args):
        engine = AlloyDBEngine.from_instance(**engine_args)
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def manager(self, engine):
        await askip_unless_model_support(engine)
        yield AlloyDBModelManager.create_sync(engine)

    async def test_create_get_list_drop_model(self, manager):
        model_id = new_model_id()
        try:
            manager.create_model(model_id, **EMBEDDING_MODEL)
            expected = dict(EMBEDDING_MODEL, model_id=model_id)
            assert asdict(manager.get_model(model_id)).items() >= expected.items()
            assert model_id in [m.model_id for m in manager.list_models()]
            manager.drop_model(model_id)
            assert manager.get_model(model_id) is None
        finally:
            manager.drop_model(model_id)
