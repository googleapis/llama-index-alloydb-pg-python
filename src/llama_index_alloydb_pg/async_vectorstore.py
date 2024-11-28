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

# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

import base64
import json
import re
import uuid
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import AlloyDBEngine


class AsyncAlloyDBVectorStore(BasePydanticVectorStore):
    """Google AlloyDB Vector Store class"""

    stores_text: bool = True
    is_embedding_query: bool = True

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncEngine,
        table_name: str,
        schema_name: str = "public",
        id_column: str = "node_id",
        text_column: str = "text",
        embedding_column: str = "embedding",
        metadata_json_column: str = "li_metadata",
        metadata_columns: List[str] = [],
        ref_doc_id_column: str = "ref_doc_id",
        node_column: str = "node",
        stores_text: bool = True,
        is_embedding_query: bool = True,
    ):
        """AsyncAlloyDBVectorStore constructor.
        Args:
            key (object): Prevent direct constructor usage.
            engine (AsyncEngine): Connection pool engine for managing connections to AlloyDB database.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str): Name of the database schema. Defaults to "public".
            id_column (str): Column that represents if of a Node. Defaults to "node_id".
            text_column (str): Column that represent text content of a Node. Defaults to "text".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the content of Node. Defaults to "embedding".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "li_metadata".
            metadata_columns (List[str]): Column(s) that represent extracted metadata keys in their own columns.
            ref_doc_id_column (str): Column that represents id of a node's parent document. Defaults to "ref_doc_id".
            node_column (str): Column that represents the whole JSON node. Defaults to "node".
            stores_text (bool): Whether the table stores text. Defaults to "True".
            is_embedding_query (bool): Whether the table query can have embeddings. Defaults to "True".


        Raises:
            Exception: If called directly by user.
        """
        if key != AsyncAlloyDBVectorStore.__create_key:
            raise Exception("Only create class through 'create' method!")

        # Delegate to Pydantic's __init__
        super().__init__(stores_text=stores_text, is_embedding_query=is_embedding_query)
        self._engine = engine
        self._table_name = table_name
        self._schema_name = schema_name
        self._id_column = id_column
        self._text_column = text_column
        self._embedding_column = embedding_column
        self._metadata_json_column = metadata_json_column
        self._metadata_columns = metadata_columns
        self._ref_doc_id_column = ref_doc_id_column
        self._node_column = node_column

    @classmethod
    async def create(
        cls: Type[AsyncAlloyDBVectorStore],
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        id_column: str = "node_id",
        text_column: str = "text",
        embedding_column: str = "embedding",
        metadata_json_column: str = "li_metadata",
        metadata_columns: List[str] = [],
        ref_doc_id_column: str = "ref_doc_id",
        node_column: str = "node",
        stores_text: bool = True,
        is_embedding_query: bool = True,
    ) -> AsyncAlloyDBVectorStore:
        """Create an AsyncAlloyDBVectorStore instance and validates the table schema.

        Args:
            engine (AlloyDBEngine): Alloy DB Engine for managing connections to AlloyDB database.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str): Name of the database schema. Defaults to "public".
            id_column (str): Column that represents if of a Node. Defaults to "node_id".
            text_column (str): Column that represent text content of a Node. Defaults to "text".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the content of Node. Defaults to "embedding".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "li_metadata".
            metadata_columns (List[str]): Column(s) that represent extracted metadata keys in their own columns.
            ref_doc_id_column (str): Column that represents id of a node's parent document. Defaults to "ref_doc_id".
            node_column (str): Column that represents the whole JSON node. Defaults to "node".
            stores_text (bool): Whether the table stores text. Defaults to "True".
            is_embedding_query (bool): Whether the table query can have embeddings. Defaults to "True".

        Raises:
            Exception: If table does not exist or follow the provided structure.

        Returns:
            AsyncAlloyDBVectorStore
        """
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{schema_name}'"
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(stmt))
            result_map = result.mappings()
            results = result_map.fetchall()
        columns = {}
        for field in results:
            columns[field["column_name"]] = field["data_type"]

        # Check columns
        if id_column not in columns:
            raise ValueError(f"Id column, {id_column}, does not exist.")
        if text_column not in columns:
            raise ValueError(f"Text column, {text_column}, does not exist.")
        text_type = columns[text_column]
        if text_type != "text" and "char" not in text_type:
            raise ValueError(
                f"Text column, {text_column}, is type, {text_type}. It must be a type of character string."
            )
        if embedding_column not in columns:
            raise ValueError(f"Embedding column, {embedding_column}, does not exist.")
        if columns[embedding_column] != "USER-DEFINED":
            raise ValueError(
                f"Embedding column, {embedding_column}, is not type Vector."
            )
        if node_column not in columns:
            raise ValueError(f"Node column, {node_column}, does not exist.")
        if columns[node_column] != "json":
            raise ValueError(f"Node column, {node_column}, is not type JSON.")
        if ref_doc_id_column not in columns:
            raise ValueError(
                f"Reference Document Id column, {ref_doc_id_column}, does not exist."
            )
        if metadata_json_column not in columns:
            raise ValueError(
                f"Metadata column, {metadata_json_column}, does not exist."
            )
        if columns[metadata_json_column] != "jsonb":
            raise ValueError(
                f"Metadata column, {metadata_json_column}, is not type JSONB."
            )
        # If using metadata_columns check to make sure column exists
        for column in metadata_columns:
            if column not in columns:
                raise ValueError(f"Metadata column, {column}, does not exist.")

        return cls(
            cls.__create_key,
            engine._pool,
            table_name,
            schema_name=schema_name,
            id_column=id_column,
            text_column=text_column,
            embedding_column=embedding_column,
            metadata_json_column=metadata_json_column,
            metadata_columns=metadata_columns,
            ref_doc_id_column=ref_doc_id_column,
            node_column=node_column,
            stores_text=stores_text,
            is_embedding_query=is_embedding_query,
        )

    @classmethod
    def class_name(cls) -> str:
        return "AsyncAlloyDBVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._engine

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Asynchronously add nodes to the table."""
        ids = []
        metadata_col_names = (
            ", " + ", ".join(self._metadata_columns)
            if len(self._metadata_columns) > 0
            else ""
        )
        metadata_col_values = (
            ", " + ", :".join(self._metadata_columns)
            if len(self._metadata_columns) > 0
            else ""
        )
        insert_stmt = f"""INSERT INTO "{self._schema_name}"."{self._table_name}"(
            {self._id_column},
            {self._text_column},
            {self._embedding_column},
            {self._metadata_json_column},
            {self._ref_doc_id_column},
            {self._node_column}
            {metadata_col_names}
        ) VALUES (:node_id, :text, :embedding, :li_metadata, :ref_doc_id, :node {metadata_col_values})
        """
        node_values_list = []
        for node in nodes:
            node_values = {
                "node_id": node.node_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "embedding": str(node.get_embedding()),
                "li_metadata": json.dumps(node.to_dict()["metadata"]),
                "ref_doc_id": node.ref_doc_id,
                "node": node.to_json(),
            }
            for metadata_column in self._metadata_columns:
                node_values[metadata_column] = node.metadata.get(metadata_column)
            node_values_list.append(node_values)
            ids.append(node.node_id)
        async with self._engine.connect() as conn:
            await conn.execute(text(insert_stmt), node_values_list)
            await conn.commit()
        return ids

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Asynchronously delete nodes belonging to provided parent document from the table."""
        query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE {self._ref_doc_id_column} = '{ref_doc_id}'"""
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronously delete a set of nodes from the table matching the provided nodes and filters."""
        # TODO: complete implementation
        return

    async def aclear(self) -> None:
        """Asynchronously delete all nodes from the table."""
        query = f'TRUNCATE TABLE "{self._schema_name}"."{self._table_name}"'
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Asynchronously get nodes from the table matching the provided nodes and filters."""
        # TODO: complete implementation
        return []

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Asynchronously query vector store."""
        # TODO: complete implementation
        return VectorStoreQueryResult()

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."
        )

    def clear(self) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."
        )

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBVectorStore. Use AlloyDBVectorStore interface instead."
        )
