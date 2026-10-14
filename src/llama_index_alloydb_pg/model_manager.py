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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

from .engine import AlloyDBEngine

MODEL_SUPPORT_FLAG = "google_ml_integration.enable_model_support"

MODEL_INFO_QUERY = (
    "SELECT model_id, model_request_url, model_provider, model_type, "
    "model_qualified_name, model_auth_type, model_auth_id, input_transform_fn, "
    "output_transform_fn FROM google_ml.model_info_view"
)

CREATE_MODEL_ENUMS = {
    "model_provider": "google_ml.model_provider",
    "model_type": "google_ml.model_type",
    "model_auth_type": "google_ml.auth_type",
}


@dataclass
class AlloyDBModel:
    """A model endpoint registered in AlloyDB AI model endpoint management."""

    model_id: str
    model_request_url: Optional[str]
    model_provider: str
    model_type: str
    model_qualified_name: Optional[str]
    model_auth_type: Optional[str]
    model_auth_id: Optional[str]
    input_transform_fn: Optional[str]
    output_transform_fn: Optional[str]


class AlloyDBModelManager:
    """Register, inspect and drop AlloyDB AI model endpoints.

    See https://cloud.google.com/alloydb/docs/ai/register-model-endpoint.
    """

    __create_key = object()

    def __init__(self, key: object, engine: AlloyDBEngine) -> None:
        """AlloyDBModelManager constructor.

        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): AlloyDB engine with a connection pool.

        Raises:
            Exception: If the constructor is called directly by the user.
        """
        if key != AlloyDBModelManager.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine

    @classmethod
    async def create(
        cls: type[AlloyDBModelManager], engine: AlloyDBEngine
    ) -> AlloyDBModelManager:
        """Create an AlloyDBModelManager after checking model endpoint support.

        Args:
            engine (AlloyDBEngine): AlloyDB engine with a connection pool.

        Returns:
            AlloyDBModelManager: A new model manager.

        Raises:
            ValueError: If model endpoint management is not available.
        """
        manager = cls(cls.__create_key, engine)
        await engine._run_as_async(manager.__avalidate())
        return manager

    @classmethod
    def create_sync(
        cls: type[AlloyDBModelManager], engine: AlloyDBEngine
    ) -> AlloyDBModelManager:
        """Synchronously create an AlloyDBModelManager. See `create`."""
        manager = cls(cls.__create_key, engine)
        engine._run_as_sync(manager.__avalidate())
        return manager

    async def __avalidate(self) -> None:
        """Raise ValueError if model endpoint management is unavailable."""
        query = """SELECT current_setting(:flag, true) AS model_support, EXISTS (
            SELECT 1 FROM pg_extension WHERE extname = 'google_ml_integration'
        ) AS installed"""
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query), {"flag": MODEL_SUPPORT_FLAG})
            row = result.mappings().one()
        if not row["installed"]:
            raise ValueError(
                "Install the google_ml_integration extension to use model endpoints."
            )
        if row["model_support"] != "on":
            raise ValueError(
                f"Set the {MODEL_SUPPORT_FLAG} database flag to on for the AlloyDB "
                "instance to enable model endpoint management."
            )

    async def __aget_model(self, model_id: str) -> Optional[AlloyDBModel]:
        """Get a registered model endpoint."""
        query = f"{MODEL_INFO_QUERY} WHERE model_id = :model_id"
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query), {"model_id": model_id})
            row = result.mappings().first()
        return AlloyDBModel(**row) if row else None

    async def aget_model(self, model_id: str) -> Optional[AlloyDBModel]:
        """Get a registered model endpoint.

        Args:
            model_id (str): The ID of the model endpoint.

        Returns:
            Optional[AlloyDBModel]: The model endpoint, or None if it is missing.
        """
        return await self._engine._run_as_async(self.__aget_model(model_id))

    def get_model(self, model_id: str) -> Optional[AlloyDBModel]:
        """Synchronously get a registered model endpoint. See `aget_model`."""
        return self._engine._run_as_sync(self.__aget_model(model_id))

    async def __alist_models(self) -> list[AlloyDBModel]:
        """List all model endpoints."""
        query = f"{MODEL_INFO_QUERY} ORDER BY model_id"
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            rows = result.mappings().fetchall()
        return [AlloyDBModel(**row) for row in rows]

    async def alist_models(self) -> list[AlloyDBModel]:
        """List all model endpoints, including the built-in ones.

        Returns:
            list[AlloyDBModel]: The model endpoints ordered by model ID.
        """
        return await self._engine._run_as_async(self.__alist_models())

    def list_models(self) -> list[AlloyDBModel]:
        """Synchronously list all model endpoints. See `alist_models`."""
        return self._engine._run_as_sync(self.__alist_models())

    async def __acreate_model(self, **args: Optional[str]) -> None:
        """Call google_ml.create_model with the arguments that are not None."""
        for name in ("model_id", "model_provider", "model_type"):
            if not args[name]:
                raise ValueError(f"{name} must be a non-empty string.")
        params = {name: value for name, value in args.items() if value is not None}
        named_args = ", ".join(
            f"{name} => CAST(:{name} AS {CREATE_MODEL_ENUMS.get(name, 'VARCHAR')})"
            for name in params
        )
        query = f"CALL google_ml.create_model({named_args})"
        async with self._engine._pool.connect() as conn:
            await conn.execute(text(query), params)
            await conn.commit()

    async def acreate_model(
        self,
        model_id: str,
        *,
        model_provider: str,
        model_type: str,
        model_qualified_name: Optional[str] = None,
        model_request_url: Optional[str] = None,
        model_auth_type: Optional[str] = None,
        model_auth_id: Optional[str] = None,
        generate_headers_fn: Optional[str] = None,
        model_in_transform_fn: Optional[str] = None,
        model_out_transform_fn: Optional[str] = None,
    ) -> None:
        """Register a model endpoint. Arguments left as None use server defaults.

        Args:
            model_id (str): A unique ID for the model endpoint.
            model_provider (str): The provider, for example "google" or "custom".
            model_type (str): The type, for example "text_embedding" or "llm".
            model_qualified_name (Optional[str]): The qualified model name.
            model_request_url (Optional[str]): The model-specific endpoint URL.
            model_auth_type (Optional[str]): The authentication type.
            model_auth_id (Optional[str]): The secret ID used for authentication.
            generate_headers_fn (Optional[str]): SQL function for custom headers.
            model_in_transform_fn (Optional[str]): SQL function for the input.
            model_out_transform_fn (Optional[str]): SQL function for the output.

        Raises:
            ValueError: If model_id, model_provider or model_type is empty.
        """
        await self._engine._run_as_async(
            self.__acreate_model(
                model_id=model_id,
                model_provider=model_provider,
                model_type=model_type,
                model_qualified_name=model_qualified_name,
                model_request_url=model_request_url,
                model_auth_type=model_auth_type,
                model_auth_id=model_auth_id,
                generate_headers_fn=generate_headers_fn,
                model_in_transform_fn=model_in_transform_fn,
                model_out_transform_fn=model_out_transform_fn,
            )
        )

    def create_model(
        self,
        model_id: str,
        *,
        model_provider: str,
        model_type: str,
        model_qualified_name: Optional[str] = None,
        model_request_url: Optional[str] = None,
        model_auth_type: Optional[str] = None,
        model_auth_id: Optional[str] = None,
        generate_headers_fn: Optional[str] = None,
        model_in_transform_fn: Optional[str] = None,
        model_out_transform_fn: Optional[str] = None,
    ) -> None:
        """Synchronously register a model endpoint. See `acreate_model`."""
        self._engine._run_as_sync(
            self.__acreate_model(
                model_id=model_id,
                model_provider=model_provider,
                model_type=model_type,
                model_qualified_name=model_qualified_name,
                model_request_url=model_request_url,
                model_auth_type=model_auth_type,
                model_auth_id=model_auth_id,
                generate_headers_fn=generate_headers_fn,
                model_in_transform_fn=model_in_transform_fn,
                model_out_transform_fn=model_out_transform_fn,
            )
        )

    async def __adrop_model(self, model_id: str) -> None:
        """Drop a model endpoint with google_ml.drop_model."""
        async with self._engine._pool.connect() as conn:
            await conn.execute(
                text("CALL google_ml.drop_model(model_id => :model_id)"),
                {"model_id": model_id},
            )
            await conn.commit()

    async def adrop_model(self, model_id: str) -> None:
        """Drop a registered model endpoint.

        Args:
            model_id (str): The ID of the model endpoint.
        """
        await self._engine._run_as_async(self.__adrop_model(model_id))

    def drop_model(self, model_id: str) -> None:
        """Synchronously drop a registered model endpoint. See `adrop_model`."""
        self._engine._run_as_sync(self.__adrop_model(model_id))
