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

from __future__ import annotations

from typing import AsyncIterable, Callable, Iterable, List, Optional

from llama_index.core.bridge.pydantic import ConfigDict
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from .async_reader import AsyncAlloyDBReader
from .engine import AlloyDBEngine

DEFAULT_METADATA_COL = "li_metadata"


class AlloyDBReader(BasePydanticReader):
    """Chat Store Table stored in an AlloyDB for PostgreSQL database."""

    __create_key = object()
    is_remote: bool = True

    def __init__(
        self,
        key: object,
        engine: AlloyDBEngine,
        reader: AsyncAlloyDBReader,
        is_remote: bool = True,
    ) -> None:
        """AlloyDBReader constructor.

        Args:
            key (object): Prevent direct constructor usage.
            engine (AlloyDBEngine): AlloyDB with pool connection to the alloydb database
            reader (AsyncAlloyDBReader): The async only AlloyDBReader implementation
            is_remote (Optional[bool]): Whether the data is loaded from a remote API or a local file.

        Raises:
            Exception: If called directly by user.
        """
        if key != AlloyDBReader.__create_key:
            raise Exception("Only create class through 'create' method!")

        super().__init__(is_remote=is_remote)

        self._engine = engine
        self.__reader = reader

    @classmethod
    async def create(
        cls: type[AlloyDBReader],
        engine: AlloyDBEngine,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        schema_name: str = "public",
        content_columns: Optional[list[str]] = None,
        metadata_columns: Optional[list[str]] = None,
        metadata_json_column: Optional[str] = None,
        format: Optional[str] = None,
        formatter: Optional[Callable] = None,
        is_remote: bool = True,
    ) -> AlloyDBReader:
        """Asynchronously create an AlloyDBReader instance.

        Args:
            engine (AlloyDBEngine): AlloyDBEngine with pool connection to the alloydb database
            query (Optional[str], optional): SQL query. Defaults to None.
            table_name (Optional[str], optional): Name of table to query. Defaults to None.
            schema_name (str, optional): Name of the schema where table is located. Defaults to "public".
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "li_metadata".
            format (Optional[str], optional): Format of page content (OneOf: text, csv, YAML, JSON). Defaults to 'text'.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.
            is_remote (Optional[bool]): Whether the data is loaded from a remote API or a local file.


        Returns:
            AlloyDBReader: A newly created instance of AlloyDBReader.
        """
        coro = AsyncAlloyDBReader.create(
            engine=engine,
            query=query,
            table_name=table_name,
            schema_name=schema_name,
            content_columns=content_columns,
            metadata_columns=metadata_columns,
            metadata_json_column=metadata_json_column,
            format=format,
            formatter=formatter,
            is_remote=is_remote,
        )
        reader = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, reader, is_remote)

    @classmethod
    def create_sync(
        cls: type[AlloyDBReader],
        engine: AlloyDBEngine,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        schema_name: str = "public",
        content_columns: Optional[list[str]] = None,
        metadata_columns: Optional[list[str]] = None,
        metadata_json_column: Optional[str] = None,
        format: Optional[str] = None,
        formatter: Optional[Callable] = None,
        is_remote: bool = True,
    ) -> AlloyDBReader:
        """Synchronously create an AlloyDBReader instance.

        Args:
            engine (AlloyDBEngine):AsyncEngine with pool connection to the alloydb database
            query (Optional[str], optional): SQL query. Defaults to None.
            table_name (Optional[str], optional): Name of table to query. Defaults to None.
            schema_name (str, optional): Name of the schema where table is located. Defaults to "public".
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "li_metadata".
            format (Optional[str], optional): Format of page content (OneOf: text, csv, YAML, JSON). Defaults to 'text'.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.
            is_remote (Optional[bool]): Whether the data is loaded from a remote API or a local file.


        Returns:
            AlloyDBReader: A newly created instance of AlloyDBReader.
        """
        coro = AsyncAlloyDBReader.create(
            engine=engine,
            query=query,
            table_name=table_name,
            schema_name=schema_name,
            content_columns=content_columns,
            metadata_columns=metadata_columns,
            metadata_json_column=metadata_json_column,
            format=format,
            formatter=formatter,
            is_remote=is_remote,
        )
        reader = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, reader, is_remote)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AlloyDBReader"

    async def aload_data(self) -> list[Document]:
        """Asynchronously load AlloyDB data into Document objects."""
        return await self._engine._run_as_async(self.__reader.aload_data())

    def load_data(self) -> list[Document]:
        """Synchronously load AlloyDB data into Document objects."""
        return self._engine._run_as_sync(self.__reader.aload_data())

    async def alazy_load_data(self) -> AsyncIterable[Document]:  # type: ignore
        """Asynchronously load AlloyDB data into Document objects lazily."""
        # The return type in the underlying base class is an Iterable which we are overriding to an AsyncIterable in this implementation.
        iterator = self.__reader.alazy_load_data().__aiter__()
        while True:
            try:
                result = await self._engine._run_as_async(iterator.__anext__())
                yield result
            except StopAsyncIteration:
                break

    def lazy_load_data(self) -> Iterable[Document]:  # type: ignore
        """Synchronously aoad AlloyDB data into Document objects lazily."""
        iterator = self.__reader.alazy_load_data().__aiter__()
        while True:
            try:
                result = self._engine._run_as_sync(iterator.__anext__())
                yield result
            except StopAsyncIteration:
                break
