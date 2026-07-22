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

import pytest
from unittest.mock import AsyncMock, MagicMock

from llama_index_alloydb_pg.model_manager import AlloyDBModel, AlloyDBModelManager


class TestAlloyDBModelManager:
    def test_model_init(self):
        model = AlloyDBModel(
            model_id="test_model",
            model_request_url="http://test.url",
            model_provider="google",
            model_type="text_embedding",
            model_qualified_name="text-embedding-004",
            model_auth_type="user_credentials",
            model_auth_id="secret_id",
            input_transform_fn="input_fn",
            output_transform_fn="output_fn",
        )
        assert model.model_id == "test_model"
        assert model.model_provider == "google"
        assert model.model_qualified_name == "text-embedding-004"

    def test_direct_init_raises(self):
        engine_mock = MagicMock()
        with pytest.raises(Exception, match="Only create class through"):
            AlloyDBModelManager(object(), engine_mock)
