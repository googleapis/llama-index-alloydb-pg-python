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
from typing import Any, Optional

import pytest
import pytest_asyncio
from llama_index.core.tools import FunctionTool
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBToolSpec

AI_FUNCTION_FLAGS = [
    "google_ml_integration.enable_ai_query_engine",
    "google_ml_integration.enable_preview_ai_functions",
]
POSITIVE_TEXT = "I love this product. It works perfectly and the support team is great."
NEGATIVE_TEXT = "Terrible product. It broke after one day and support never answered."
ARTICLE = (
    "AlloyDB for PostgreSQL is a fully managed, PostgreSQL-compatible database "
    "service from Google Cloud. It separates compute from storage, and its storage "
    "layer is built on Google's distributed infrastructure, which lets it scale reads "
    "with read pool instances. It offers high availability with automatic failover, "
    "continuous backups and point-in-time recovery. Its columnar engine keeps "
    "frequently queried columns in memory to speed up analytical queries without "
    "changing the schema. AlloyDB AI adds vector search with the ScaNN index and lets "
    "applications call machine learning models, such as Gemini, directly from SQL to "
    "generate embeddings, analyze sentiment, summarize text and evaluate conditions "
    "in natural language. Developers can migrate existing PostgreSQL applications "
    "without code changes and use familiar tools and extensions."
)
TRUE_CONDITION = "Is 2 + 2 equal to 4?"
FALSE_CONDITION = "Is 2 + 2 equal to 5?"
SUMMARY_INSTRUCTIONS = "Start with the word SUMMARY: and use at most 25 words."
UNKNOWN_MODEL_ID = "no_such_model_" + uuid.uuid4().hex


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def arun_sql(
    engine: AlloyDBEngine, query: str, params: Optional[dict[str, Any]] = None
) -> Any:
    async def run() -> Any:
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query), params or {})
            value = result.mappings().one()["value"] if result.returns_rows else None
            await conn.commit()
        return value

    return await engine._run_as_async(run())


async def askip_if_flags_off(engine: AlloyDBEngine, flags: list[str]) -> None:
    query = "SELECT current_setting(:flag, true) AS value"
    for flag in flags:
        if await arun_sql(engine, query, {"flag": flag}) != "on":
            pytest.skip(f"A required database flag is off: {flag}.")


def check_instructed_summary(summary: str) -> None:
    # google_ml.summarize often wraps its output in double quotes.
    assert summary.removeprefix('"').upper().startswith("SUMMARY")
    assert "alloydb" in summary.lower()
    assert len(summary) < len(ARTICLE) / 2


@pytest.fixture(scope="module")
def engine_args() -> dict[str, str]:
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
class TestAlloyDBToolSpecAsync:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self, engine_args):
        engine = await AlloyDBEngine.afrom_instance(**engine_args)
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def ai_flags(self, engine):
        await askip_if_flags_off(engine, AI_FUNCTION_FLAGS)

    @pytest_asyncio.fixture(scope="class")
    async def tool_spec(self, engine, ai_flags):
        return AlloyDBToolSpec(engine)

    @pytest_asyncio.fixture(scope="class")
    async def null_model_id(self, engine):
        """Registers a model whose output transform always returns NULL."""
        await askip_if_flags_off(engine, ["google_ml_integration.enable_model_support"])
        suffix = uuid.uuid4().hex[:8]
        model_id, transform = f"null_llm_{suffix}", f"public.null_llm_output_{suffix}"
        await arun_sql(
            engine,
            f"CREATE FUNCTION {transform}(model_id VARCHAR, response JSON)"
            " RETURNS TEXT LANGUAGE sql AS 'SELECT NULL::text'",
        )
        try:
            await arun_sql(
                engine,
                """CALL google_ml.create_model(
                    model_id => :model_id,
                    model_request_url => 'publishers/google/models/'
                        || current_setting(:default_model) || '\\:generateContent',
                    model_provider => 'google',
                    model_type => 'llm',
                    model_qualified_name => current_setting(:default_model),
                    model_in_transform_fn => 'google_ml.gemini_llm_input_transform',
                    model_out_transform_fn => :transform)""",
                {
                    "model_id": model_id,
                    "default_model": "google_ml_integration.default_llm_model",
                    "transform": transform,
                },
            )
            try:
                yield model_id
            finally:
                await arun_sql(
                    engine, "CALL google_ml.drop_model(:id)", {"id": model_id}
                )
        finally:
            await arun_sql(engine, f"DROP FUNCTION {transform}(VARCHAR, JSON)")

    async def test_aanalyze_sentiment(self, tool_spec):
        assert await tool_spec.aanalyze_sentiment(POSITIVE_TEXT) == "positive"
        assert await tool_spec.aanalyze_sentiment(NEGATIVE_TEXT) == "negative"

    async def test_asummarize(self, tool_spec):
        assert "alloydb" in (await tool_spec.asummarize(ARTICLE)).lower()
        summary = await tool_spec.asummarize(ARTICLE, instructions=SUMMARY_INSTRUCTIONS)
        check_instructed_summary(summary)

    async def test_aevaluate_condition(self, tool_spec):
        assert await tool_spec.aevaluate_condition(TRUE_CONDITION) is True
        assert await tool_spec.aevaluate_condition(FALSE_CONDITION) is False

    async def test_to_tool_list(self, tool_spec):
        tools = tool_spec.to_tool_list()
        names = [tool.metadata.name for tool in tools]
        assert names == ["analyze_sentiment", "summarize", "evaluate_condition"]
        for tool in tools:
            assert isinstance(tool, FunctionTool)
            assert tool.async_fn == getattr(tool_spec, f"a{tool.metadata.name}")
        sentiment, summarize, condition = tools
        assert "positive, negative or neutral" in sentiment.metadata.description
        schema = summarize.metadata.get_parameters_dict()
        assert list(schema["properties"]) == ["text", "instructions"]
        assert schema["required"] == ["text"]
        output = await sentiment.acall(text=NEGATIVE_TEXT)
        assert output.raw_output == "negative"
        output = await condition.acall(condition=TRUE_CONDITION)
        assert output.raw_output is True

    async def test_blank_input(self, engine):
        spec = AlloyDBToolSpec(engine)
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            await spec.aanalyze_sentiment("   ")
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            await spec.asummarize("")
        with pytest.raises(ValueError, match="condition must be a non-empty string"):
            await spec.aevaluate_condition(" \n ")
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            AlloyDBToolSpec(engine, model_id="  ")

    async def test_unknown_model_id_raises(self, engine, ai_flags):
        spec = AlloyDBToolSpec(engine, model_id=UNKNOWN_MODEL_ID)
        with pytest.raises(DBAPIError) as exc_info:
            await spec.aanalyze_sentiment(POSITIVE_TEXT)
        assert exc_info.value.orig.sqlstate  # GAV03 on AlloyDB with PostgreSQL 17
        assert UNKNOWN_MODEL_ID in str(exc_info.value.orig)

    async def test_null_result_raises(self, engine, ai_flags, null_model_id):
        spec = AlloyDBToolSpec(engine, model_id=null_model_id)
        with pytest.raises(
            ValueError, match="google_ml.analyze_sentiment returned NULL"
        ):
            await spec.aanalyze_sentiment(POSITIVE_TEXT)
        with pytest.raises(ValueError, match="google_ml.summarize returned NULL"):
            await spec.asummarize(ARTICLE, instructions=SUMMARY_INSTRUCTIONS)
        with pytest.raises(ValueError, match="google_ml.if returned NULL"):
            await spec.aevaluate_condition(TRUE_CONDITION)


@pytest.mark.asyncio(loop_scope="class")
class TestAlloyDBToolSpecSync:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self, engine_args):
        engine = AlloyDBEngine.from_instance(**engine_args)
        yield engine
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def tool_spec(self, engine):
        await askip_if_flags_off(engine, AI_FUNCTION_FLAGS)
        return AlloyDBToolSpec(engine)

    async def test_analyze_sentiment(self, tool_spec):
        assert tool_spec.analyze_sentiment(POSITIVE_TEXT) == "positive"
        sentiment = tool_spec.to_tool_list()[0]
        assert sentiment.call(text=NEGATIVE_TEXT).raw_output == "negative"

    async def test_summarize(self, tool_spec):
        summary = tool_spec.summarize(ARTICLE, instructions=SUMMARY_INSTRUCTIONS)
        check_instructed_summary(summary)

    async def test_evaluate_condition(self, tool_spec):
        assert tool_spec.evaluate_condition(TRUE_CONDITION) is True
        condition = tool_spec.to_tool_list()[2]
        assert condition.call(condition=FALSE_CONDITION).raw_output is False
