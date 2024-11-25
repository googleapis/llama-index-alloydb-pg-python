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

import json
from typing import Dict, List, Optional, Sequence, Tuple

from llama_index.core.constants import DATA_KEY
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.storage.index_store.utils import (
    index_struct_to_json,
    json_to_index_struct,
)
from llama_index.core.storage.kvstore.types import DEFAULT_BATCH_SIZE
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import AlloyDBEngine


class AsyncAlloyDBIndexStore(BaseIndexStore):
    """Index Store Table stored in an AlloyDB for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncEngine,
        table_name: str,
        schema_name: str = "public",
        batch_size: str = DEFAULT_BATCH_SIZE,
    ):
        """AsyncAlloyDBIndexStore constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (AlloyDBEngine): Database connection pool.
            table_name (str): Table name that stores the index metadata.
            schema_name (str): The schema name where the table is located. Defaults to "public"
            batch_size (str): The default batch size for bulk inserts. Defaults to 1.

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != AsyncAlloyDBIndexStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self._table_name = table_name
        self._schema_name = schema_name
        self._batch_size = batch_size

    @classmethod
    async def create(
        cls,
        engine: AlloyDBEngine,
        table_name: str,
        schema_name: str = "public",
        batch_size: str = DEFAULT_BATCH_SIZE,
    ) -> AsyncAlloyDBIndexStore:
        """Create a new AsyncAlloyDBIndexStore instance.

        Args:
            engine (AlloyDBEngine): AlloyDB engine to use.
            table_name (str): Table name that stores the index metadata.
            schema_name (str): The schema name where the table is located. Defaults to "public"
            batch_size (str): The default batch size for bulk inserts. Defaults to 1.

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AsyncAlloyDBIndexStore: A newly created instance of AsyncAlloyDBIndexStore.
        """
        table_schema = await engine._aload_table_schema(table_name, schema_name)
        column_names = table_schema.columns.keys()

        required_columns = ["index_id", "type", "index_data"]

        if not (all(x in column_names for x in required_columns)):
            raise IndexError(
                f"Table '{schema_name}'.'{table_name}' has incorrect schema. Got "
                f"column names '{column_names}' but required column names "
                f"'{required_columns}'.\nPlease create table with following schema:"
                f"\nCREATE TABLE {schema_name}.{table_name} ("
                "\n    index_id VARCHAR PRIMARY KEY,"
                "\n    type VARCHAR NOT NULL,"
                "\n    index_data JSONB NOT NULL"
                "\n);"
            )

        return cls(cls.__create_key, engine._pool, table_name, schema_name, batch_size)

    async def _aexecute_query(self, query, params=None):
        async with self._engine.connect() as conn:
            await conn.execute(text(query), params)
            await conn.commit()

    async def _afetch_query(self, query):
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
            await conn.commit()
        return results

    async def _get_all_from_table(
        self,
    ) -> Optional[Dict[str, str, dict]]:
        """Gets all the rows from the index store.

        Returns:
            Optional[
                Dict[
                  str,    # Index_id
                  str,    # Type
                  dict    # Index_data
                ]
            ]
        """
        query = f"""SELECT * from "{self._schema_name}"."{self._table_name}";"""
        results = await self._afetch_query(query)
        if results:
            return results
        return None

    async def _get_from_table(
        self, index_id: str, columns: str = "*"
    ) -> Optional[dict]:
        """Gets the specific rows from the index store with the provided index_id.

        Args:
            index_id (str): The index_id to fetch the index
            columns (str): Column to be returned in the query. Defaults to * (all)

        Returns:
            Optional[
              Dict : Dictionary with the column name as key and data from the row as value.
            ]
        """
        query = f"""SELECT {columns} from "{self._schema_name}"."{self._table_name}" WHERE index_id = '{index_id}';"""
        result = await self._afetch_query(query)
        if result:
            return result[0]
        return None

    async def _put_all_to_table(
        self,
        rows: List[Tuple[str, str, dict]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Puts a list of rows into the index store table.

        Args:
            rows (List[Tuple[str, str, str, dict]]): List of tuples of the row(index_id, type, index_data)
            batch_size (int): batch_size to insert the rows. Defaults to 1.

        Returns:
            None
        """
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            # Prepare the VALUES part of the SQL statement
            values_clause = ", ".join(
                f"(:index_id_{i}, :type_{i}, :index_data_{i})"
                for i, _ in enumerate(batch)
            )

            # Insert statement
            stmt = f"""
              INSERT INTO "{self._schema_name}"."{self._table_name}" (index_id, type, index_data)
              VALUES {values_clause}
              ON CONFLICT (index_id)
              DO UPDATE SET
              type = EXCLUDED.type,
              index_data = EXCLUDED.index_data;
              """

            params = {}
            for i, (index_id, type, index_data) in enumerate(batch):
                params[f"index_id_{i}"] = index_id
                params[f"type_{i}"] = type
                params[f"index_data_{i}"] = json.dumps(index_data)

            await self._aexecute_query(stmt, params)

    async def _put_to_table(
        self,
        index_id: str,
        type: str,
        index_data: dict,
    ) -> None:
        """Puts a row into the index store table.

        Args:
            index_id (str): index id.
            type (str): type of index.
            index_data (dict): Dictionary containing IndexStruct data.

        Returns:
            None
        """
        await self._put_all_to_table([(index_id, type, index_data)])

    async def _delete_from_table(self, index_id: str) -> None:
        """Delete a value from the store.

        Args:
            index_id (str): index_id to be deleted

        Returns:
            List of deleted rows.
        """
        query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE index_id = '{index_id}' RETURNING *; """
        result = await self._afetch_query(query)
        return result

    async def aindex_structs(self) -> List[IndexStruct]:
        index_list = await self._get_all_from_table()
        return {
            index["index_id"]: json_to_index_struct(index["index_data"])
            for index in index_list
        }

    async def aadd_index_struct(self, index_struct: IndexStruct) -> None:
        """Add an index struct.

        Args:
            index_struct (IndexStruct): index struct

        """
        key = index_struct.index_id
        data = index_struct_to_json(index_struct)
        type = index_struct.get_type()
        await self._put_to_table(index_id=key, type=type, index_data=data)

    async def adelete_index_struct(self, key: str) -> None:
        await self._delete_from_table(index_id=key)

    async def aget_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        if struct_id is None:
            structs = await self.aindex_structs()
            assert len(structs) == 1
            return structs[0]
        else:
            json = await self._get_from_table(index_id=struct_id)
            if json is None:
                return None
            return json_to_index_struct(json.get("index_data"))

    def index_structs(self) -> List[IndexStruct]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBIndexStore . Use AlloyDBIndexStore  interface instead."
        )

    def add_index_struct(self, index_struct: IndexStruct) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBIndexStore . Use AlloyDBIndexStore  interface instead."
        )

    def delete_index_struct(self, key: str) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBIndexStore . Use AlloyDBIndexStore  interface instead."
        )

    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncAlloyDBIndexStore . Use AlloyDBIndexStore  interface instead."
        )
