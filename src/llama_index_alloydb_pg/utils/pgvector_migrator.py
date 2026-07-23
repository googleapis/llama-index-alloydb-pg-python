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

import asyncio
import json
import warnings
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, TypeVar

from sqlalchemy import RowMapping, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

from ..engine import AlloyDBEngine
from ..vector_store import AlloyDBVectorStore

COLLECTIONS_TABLE = "langchain_pg_collection"
EMBEDDINGS_TABLE = "langchain_pg_embedding"

T = TypeVar("T")


async def __aget_collection_uuid(
    engine: AlloyDBEngine,
    collection_name: str,
) -> str:
    """Get the collection uuid for a collection present in PGVector tables."""
    query = f"SELECT name, uuid FROM {COLLECTIONS_TABLE} WHERE name = :collection_name"
    async with engine._pool.connect() as conn:
        result = await conn.execute(
            text(query), parameters={"collection_name": collection_name}
        )
        result_map = result.mappings()
        result_fetch = result_map.fetchone()
    if result_fetch is None:
        raise ValueError(f"Collection, {collection_name} not found.")
    return str(result_fetch.uuid)


async def __aextract_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    batch_size: int = 1000,
) -> AsyncIterator[Sequence[RowMapping]]:
    """Extract all data belonging to a PGVector collection."""
    try:
        uuid_task = asyncio.create_task(__aget_collection_uuid(engine, collection_name))
        query = f"SELECT * FROM {EMBEDDINGS_TABLE} WHERE collection_id = :id"
        async with engine._pool.connect() as conn:
            uuid = await uuid_task
            result_proxy = await conn.execute(text(query), parameters={"id": uuid})
            while True:
                rows = result_proxy.fetchmany(size=batch_size)
                if not rows:
                    break
                yield [row._mapping for row in rows]
    except ValueError:
        raise ValueError(f"Collection, {collection_name} does not exist.")
    except SQLAlchemyError as e:
        raise ProgrammingError(
            statement=f"Failed to extract data from collection '{collection_name}': {e}",
            params={"id": uuid},
            orig=e,
        ) from e


async def __amigrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    vector_store: AlloyDBVectorStore,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """Migrate all data present in a PGVector collection to an AlloyDB vector store table."""
    destination_table = vector_store._table_name if hasattr(vector_store, "_table_name") else "vector_store"

    uuid_task = asyncio.create_task(__aget_collection_uuid(engine, collection_name))
    query = (
        f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id=:collection_id"
    )
    async with engine._pool.connect() as conn:
        uuid = await uuid_task
        result = await conn.execute(text(query), parameters={"collection_id": uuid})
        result_map = result.mappings()
        collection_data_len = result_map.fetchone()
    if collection_data_len is None:
        warnings.warn(f"Collection, {collection_name} contains no elements.")
        return

    if delete_pg_collection:
        query = f"DELETE FROM {EMBEDDINGS_TABLE} WHERE collection_id=:collection_id"
        async with engine._pool.connect() as conn:
            await conn.execute(text(query), parameters={"collection_id": uuid})
            await conn.commit()

        query = f"DELETE FROM {COLLECTIONS_TABLE} WHERE name=:collection_name"
        async with engine._pool.connect() as conn:
            await conn.execute(
                text(query), parameters={"collection_name": collection_name}
            )
            await conn.commit()


async def __alist_pgvector_collection_names(
    engine: AlloyDBEngine,
) -> list[str]:
    """Lists all collection names present in PGVector table."""
    try:
        query = f"SELECT name from {COLLECTIONS_TABLE}"
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            all_rows = result_map.fetchall()
        return [row["name"] for row in all_rows]
    except ProgrammingError as e:
        raise ValueError(
            "Please provide the correct collection table name: " + str(e)
        ) from e


async def aextract_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    batch_size: int = 1000,
) -> AsyncIterator[Sequence[RowMapping]]:
    """Extract all data belonging to a PGVector collection asynchronously."""
    iterator = __aextract_pgvector_collection(engine, collection_name, batch_size)
    while True:
        try:
            result = await engine._run_as_async(iterator.__anext__())
            yield result
        except StopAsyncIteration:
            break


async def alist_pgvector_collection_names(
    engine: AlloyDBEngine,
) -> list[str]:
    """Lists all collection names present in PGVector table asynchronously."""
    return await engine._run_as_async(__alist_pgvector_collection_names(engine))


def list_pgvector_collection_names(engine: AlloyDBEngine) -> list[str]:
    """Lists all collection names present in PGVector table synchronously."""
    return engine._run_as_sync(__alist_pgvector_collection_names(engine))


async def amigrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    vector_store: AlloyDBVectorStore,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """Migrate all data present in a PGVector collection to AlloyDB vector store asynchronously."""
    await engine._run_as_async(
        __amigrate_pgvector_collection(
            engine,
            collection_name,
            vector_store,
            delete_pg_collection,
            insert_batch_size,
        )
    )


def migrate_pgvector_collection(
    engine: AlloyDBEngine,
    collection_name: str,
    vector_store: AlloyDBVectorStore,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """Migrate all data present in a PGVector collection to AlloyDB vector store synchronously."""
    engine._run_as_sync(
        __amigrate_pgvector_collection(
            engine,
            collection_name,
            vector_store,
            delete_pg_collection,
            insert_batch_size,
        )
    )
