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

from unittest.mock import MagicMock

from llama_index_alloydb_pg.embeddings import AlloyDBEmbedding


class TestAlloyDBEmbedding:
    def test_embed_query_inline(self):
        engine_mock = MagicMock()
        emb = AlloyDBEmbedding(engine=engine_mock, model_id="text-embedding-004")
        assert emb.embed_query_inline("hello world") == "embedding('text-embedding-004', 'hello world')::vector"
