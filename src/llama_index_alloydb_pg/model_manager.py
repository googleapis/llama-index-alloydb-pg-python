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

from __future__ import annotations

from typing import Any, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from .engine import AlloyDBEngine


class AlloyDBModel:
    def __init__(
        self,
        model_id: str,
        model_request_url: Optional[str],
        model_provider: str,
        model_type: str,
        model_qualified_name: str,
        model_auth_type: Optional[str],
        model_auth_id: Optional[str],
        input_transform_fn: Optional[str],
        output_transform_fn: Optional[str],
        generate_headers_fn: Optional[str] = None,
        input_batch_transform_fn: Optional[str] = None,
        output_batch_transform_fn: Optional[str] = None,
        model_availability: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.model_request_url = model_request_url
        self.model_provider = model_provider
        self.model_type = model_type
        self.model_qualified_name = model_qualified_name
        self.model_auth_type = model_auth_type
        self.model_auth_id = model_auth_id
        self.input_transform_fn = input_transform_fn
        self.output_transform_fn = output_transform_fn
        self.generate_headers_fn = generate_headers_fn or kwargs.get("header_gen_fn")
        self.input_batch_transform_fn = input_batch_transform_fn
        self.output_batch_transform_fn = output_batch_transform_fn
        self.model_availability = model_availability


class AlloyDBModelManager:
    """Manage models to be used with google_ml_integration Extension in AlloyDB.
    Refer to Model Endpoint Management: https://cloud.google.com/alloydb/docs/ai/model-endpoint-overview
    """

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AlloyDBEngine,
    ):
        if key != AlloyDBModelManager.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine

    @classmethod
    async def create(
        cls: type[AlloyDBModelManager],
        engine: AlloyDBEngine,
    ) -> AlloyDBModelManager:
        manager = AlloyDBModelManager(cls.__create_key, engine)
        coro = manager.__avalidate()
        await engine._run_as_async(coro)
        return manager

    @classmethod
    def create_sync(
        cls: type[AlloyDBModelManager],
        engine: AlloyDBEngine,
    ) -> AlloyDBModelManager:
        manager = AlloyDBModelManager(cls.__create_key, engine)
        coro = manager.__avalidate()
        engine._run_as_sync(coro)
        return manager

    async def aget_model(self, model_id: str) -> Optional[AlloyDBModel]:
        """Lists the model details for a specific model_id asynchronously."""
        result = await self._engine._run_as_async(self.__aget_model(model_id=model_id))
        return result

    def get_model(self, model_id: str) -> Optional[AlloyDBModel]:
        """Lists the model details for a specific model_id synchronously."""
        return self._engine._run_as_sync(self.aget_model(model_id=model_id))

    async def alist_models(self) -> list[AlloyDBModel]:
        """Lists all the models and its details asynchronously."""
        results = await self._engine._run_as_async(self.__alist_models())
        return results

    def list_models(self) -> list[AlloyDBModel]:
        """Lists all the models and its details synchronously."""
        return self._engine._run_as_sync(self.alist_models())

    async def acreate_model(
        self,
        model_id: str,
        model_provider: str,
        model_type: str,
        model_qualified_name: str,
        **kwargs: dict[str, str],
    ) -> None:
        """Creates a registration for a model endpoint asynchronously."""
        await self._engine._run_as_async(
            self.__acreate_model(
                model_id,
                model_provider,
                model_type,
                model_qualified_name,
                **kwargs,
            )
        )

    def create_model(
        self,
        model_id: str,
        model_provider: str,
        model_type: str,
        model_qualified_name: str,
        **kwargs: dict[str, str],
    ) -> None:
        """Creates a registration for a model endpoint synchronously."""
        self._engine._run_as_sync(
            self.acreate_model(
                model_id,
                model_provider,
                model_type,
                model_qualified_name,
                **kwargs,
            )
        )

    async def adrop_model(self, model_id: str) -> None:
        """Removes an already registered model asynchronously."""
        await self._engine._run_as_async(self.__adrop_model(model_id))

    def drop_model(self, model_id: str) -> None:
        """Removes an already registered model synchronously."""
        self._engine._run_as_sync(self.adrop_model(model_id))

    async def __avalidate(self) -> None:
        extension_version = await self.__fetch_google_ml_extension()
        db_flag = await self.__fetch_db_flag()
        if extension_version < "1.5.3":
            raise Exception(
                "Please upgrade google_ml_integration EXTENSION to version 1.5.3 or above."
            )
        if db_flag != "on":
            raise Exception(
                "google_ml_integration.enable_model_support DB Flag not set."
            )

    async def __query_db(self, query: str) -> Sequence[RowMapping]:
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
        return results

    async def __aget_model(self, model_id: str) -> Optional[AlloyDBModel]:
        query = f"""SELECT * FROM
                google_ml.list_model('{model_id}')
                AS t(model_id VARCHAR,
                model_availability text,
                model_request_url VARCHAR,
                model_provider google_ml.model_provider,
                model_type google_ml.model_type,
                model_qualified_name VARCHAR,
                model_auth_type google_ml.auth_type,
                model_auth_id VARCHAR,
                header_gen_fn VARCHAR,
                input_transform_fn VARCHAR,
                output_transform_fn VARCHAR,
                input_batch_transform_fn VARCHAR,
                output_batch_transform_fn VARCHAR)"""

        try:
            result = await self.__query_db(query)
        except Exception:
            return None
        if not result:
            return None
        return self.__convert_dict_to_dataclass(result)[0]

    async def __alist_models(self) -> list[AlloyDBModel]:
        query = "SELECT * FROM google_ml.model_info_view;"
        result = await self.__query_db(query)
        return self.__convert_dict_to_dataclass(result)

    async def __acreate_model(
        self,
        model_id: str,
        model_provider: str,
        model_type: str,
        model_qualified_name: str,
        **kwargs: dict[str, str],
    ) -> None:
        query = f"""
        CALL
        google_ml.create_model(
        model_id => '{model_id}',
        model_provider => '{model_provider}',
        model_type => '{model_type}',
        model_qualified_name => '{model_qualified_name}',"""
        for key, value in kwargs.items():
            query = query + f" {key} => '{value}',"
        query = query.strip(",") + ");"
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def __adrop_model(self, model_id: str) -> None:
        query = f"CALL google_ml.drop_model('{model_id}');"
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def __fetch_google_ml_extension(self) -> str:
        create_extension_query = """
        DO $$
        BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM pg_extension WHERE extname = 'google_ml_integration' )
          THEN CREATE EXTENSION google_ml_integration VERSION '1.5.3' CASCADE;
        END IF;
        END
        $$;
        """
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(create_extension_query))
            await conn.commit()
        extension_version_query = "SELECT extversion FROM pg_extension WHERE extname = 'google_ml_integration';"
        result = await self.__query_db(extension_version_query)
        return result[0]["extversion"]

    async def __fetch_db_flag(self) -> str:
        db_flag_query = "SELECT setting FROM pg_settings where name = 'google_ml_integration.enable_model_support';"
        result = await self.__query_db(db_flag_query)
        return result[0]["setting"]

    def __convert_dict_to_dataclass(
        self, list_of_rows: Sequence[RowMapping]
    ) -> list[AlloyDBModel]:
        return [AlloyDBModel(**row) for row in list_of_rows]
