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

from typing import Any, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from sqlalchemy import text as sql_text

from .engine import AlloyDBEngine

_SENTIMENT_QUERY = "SELECT google_ml.analyze_sentiment(CAST(:text AS TEXT)) AS result"
_SENTIMENT_MODEL_QUERY = (
    "SELECT google_ml.analyze_sentiment(CAST(:text AS TEXT),"
    " CAST(:model_id AS VARCHAR)) AS result"
)
_SUMMARIZE_QUERY = (
    "SELECT google_ml.summarize(CAST(:text AS TEXT),"
    " CAST(:instructions AS TEXT)) AS result"
)
_SUMMARIZE_MODEL_QUERY = (
    "SELECT google_ml.summarize(CAST(:text AS TEXT),"
    " CAST(:instructions AS TEXT), CAST(:model_id AS VARCHAR)) AS result"
)
_IF_QUERY = "SELECT google_ml.if(CAST(:condition AS TEXT)) AS result"
_IF_MODEL_QUERY = (
    "SELECT google_ml.if(CAST(:condition AS TEXT),"
    " CAST(:model_id AS VARCHAR)) AS result"
)


class AlloyDBToolSpec(BaseToolSpec):
    """Tools that run AlloyDB AI functions in the database.

    ``AlloyDBToolSpec(engine).to_tool_list()`` returns one ``FunctionTool``
    per function, each with a native async path, ready for an agent such as
    ``FunctionAgent``. Requires AlloyDB with the ``google_ml_integration``
    extension and its AI function flags turned on.

    Args:
        engine (AlloyDBEngine): Connection pool to the AlloyDB database.
        model_id (Optional[str]): ID of the model the AI functions use. If
            None, the database uses the model named by its
            ``google_ml_integration.default_llm_model`` flag. Defaults to None.

    Raises:
        ValueError: If ``model_id`` is empty or blank.
    """

    spec_functions = [
        ("analyze_sentiment", "aanalyze_sentiment"),
        ("summarize", "asummarize"),
        ("evaluate_condition", "aevaluate_condition"),
    ]

    def __init__(self, engine: AlloyDBEngine, model_id: Optional[str] = None) -> None:
        if model_id is not None and not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        self._engine = engine
        self._model_id = model_id

    async def __afetch(
        self, function: str, query: str, model_query: str, params: dict[str, Any]
    ) -> Any:
        """Run an AI function query and return its result, which can't be NULL."""
        if self._model_id is not None:
            query = model_query
            params = {**params, "model_id": self._model_id}
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(sql_text(query), params)
            value = result.mappings().one()["result"]
        if value is None:
            raise ValueError(f"google_ml.{function} returned NULL.")
        return value

    async def __aanalyze_sentiment(self, text: str) -> str:
        if not text.strip():
            raise ValueError("text must be a non-empty string.")
        return await self.__afetch(
            "analyze_sentiment",
            _SENTIMENT_QUERY,
            _SENTIMENT_MODEL_QUERY,
            {"text": text},
        )

    async def __asummarize(self, text: str, instructions: Optional[str]) -> str:
        if not text.strip():
            raise ValueError("text must be a non-empty string.")
        return await self.__afetch(
            "summarize",
            _SUMMARIZE_QUERY,
            _SUMMARIZE_MODEL_QUERY,
            {"text": text, "instructions": instructions},
        )

    async def __aevaluate_condition(self, condition: str) -> bool:
        if not condition.strip():
            raise ValueError("condition must be a non-empty string.")
        return await self.__afetch(
            "if", _IF_QUERY, _IF_MODEL_QUERY, {"condition": condition}
        )

    def analyze_sentiment(self, text: str) -> str:
        """Classify the sentiment of a text as positive, negative or neutral.

        Args:
            text (str): The text to analyze, such as a review or a message.

        Returns:
            str: "positive", "negative" or "neutral".

        Raises:
            ValueError: If ``text`` is blank or the model gives no answer.
        """
        return self._engine._run_as_sync(self.__aanalyze_sentiment(text))

    async def aanalyze_sentiment(self, text: str) -> str:
        """Classify the sentiment of a text as positive, negative or neutral.

        Args:
            text (str): The text to analyze, such as a review or a message.

        Returns:
            str: "positive", "negative" or "neutral".

        Raises:
            ValueError: If ``text`` is blank or the model gives no answer.
        """
        return await self._engine._run_as_async(self.__aanalyze_sentiment(text))

    def summarize(self, text: str, instructions: Optional[str] = None) -> str:
        """Summarize a text, such as an article or a set of reviews.

        Args:
            text (str): The text to summarize.
            instructions (Optional[str]): Extra instructions for the summary,
                for example "Use three bullet points". Defaults to None.

        Returns:
            str: The summary.

        Raises:
            ValueError: If ``text`` is blank or the model gives no answer.
        """
        return self._engine._run_as_sync(self.__asummarize(text, instructions))

    async def asummarize(self, text: str, instructions: Optional[str] = None) -> str:
        """Summarize a text, such as an article or a set of reviews.

        Args:
            text (str): The text to summarize.
            instructions (Optional[str]): Extra instructions for the summary,
                for example "Use three bullet points". Defaults to None.

        Returns:
            str: The summary.

        Raises:
            ValueError: If ``text`` is blank or the model gives no answer.
        """
        return await self._engine._run_as_async(self.__asummarize(text, instructions))

    def evaluate_condition(self, condition: str) -> bool:
        """Decide whether a natural-language condition is true or false.

        Args:
            condition (str): A yes or no question or a statement that includes
                the text it is about, for example "Does the review 'The
                battery died in a day' complain about the battery?".

        Returns:
            bool: True if the condition holds, otherwise False.

        Raises:
            ValueError: If ``condition`` is blank or the model gives no answer.
        """
        return self._engine._run_as_sync(self.__aevaluate_condition(condition))

    async def aevaluate_condition(self, condition: str) -> bool:
        """Decide whether a natural-language condition is true or false.

        Args:
            condition (str): A yes or no question or a statement that includes
                the text it is about, for example "Does the review 'The
                battery died in a day' complain about the battery?".

        Returns:
            bool: True if the condition holds, otherwise False.

        Raises:
            ValueError: If ``condition`` is blank or the model gives no answer.
        """
        return await self._engine._run_as_async(self.__aevaluate_condition(condition))
