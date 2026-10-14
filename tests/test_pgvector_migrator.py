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
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index_alloydb_pg.utils.pgvector_migrator import (
    alist_pgvector_collection_names,
    list_pgvector_collection_names,
)


class TestPGVectorMigratorUnit:
    def test_functions_exist(self):
        assert callable(list_pgvector_collection_names)
        assert callable(alist_pgvector_collection_names)
