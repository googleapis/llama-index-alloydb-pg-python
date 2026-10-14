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

import enum
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyMixin:
    operator: str
    search_function: str
    index_function: str
    scann_index_function: str


class DistanceStrategy(StrategyMixin, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "<->", "l2_distance", "vector_l2_ops", "l2"
    COSINE_DISTANCE = "<=>", "cosine_distance", "vector_cosine_ops", "cosine"
    INNER_PRODUCT = "<#>", "inner_product", "vector_ip_ops", "dot_product"


DEFAULT_DISTANCE_STRATEGY: DistanceStrategy = DistanceStrategy.COSINE_DISTANCE
DEFAULT_INDEX_NAME_SUFFIX: str = "li_vectorindex"

# ScaNN index modes, from https://cloud.google.com/alloydb/docs/ai/create-scann-index
_SCANN_MODES = ("AUTO", "MANUAL")
# Lower bound of the scann.pct_leaves_to_search percentage, from
# https://cloud.google.com/alloydb/docs/reference/ai/scann-index-reference
_MIN_PCT_LEAVES_TO_SEARCH = 0
# Upper bound of the scann.pct_leaves_to_search percentage, from
# https://cloud.google.com/alloydb/docs/reference/ai/scann-index-reference
_MAX_PCT_LEAVES_TO_SEARCH = 100


@dataclass
class BaseIndex(ABC):
    name: Optional[str] = None
    index_type: str = "base"
    distance_strategy: DistanceStrategy = field(
        default_factory=lambda: DistanceStrategy.COSINE_DISTANCE
    )
    partial_indexes: Optional[list[str]] = None

    @abstractmethod
    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        raise NotImplementedError(
            "index_options method must be implemented by subclass"
        )


@dataclass
class ExactNearestNeighbor(BaseIndex):
    index_type: str = "exactnearestneighbor"


@dataclass
class QueryOptions(ABC):
    @abstractmethod
    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        raise NotImplementedError("to_parameter method must be implemented by subclass")

    @abstractmethod
    def to_string(self) -> str:
        """Convert index attributes to string."""
        raise NotImplementedError("to_string method must be implemented by subclass")


@dataclass
class HNSWIndex(BaseIndex):
    index_type: str = "hnsw"
    m: int = 16
    ef_construction: int = 64

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        return f"(m = {self.m}, ef_construction = {self.ef_construction})"


@dataclass
class HNSWQueryOptions(QueryOptions):
    ef_search: int = 40

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        return [f"hnsw.ef_search = {self.ef_search}"]

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return f"hnsw.ef_search = {self.ef_search}"


@dataclass
class IVFFlatIndex(BaseIndex):
    index_type: str = "ivfflat"
    lists: int = 100

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        return f"(lists = {self.lists})"


@dataclass
class IVFFlatQueryOptions(QueryOptions):
    probes: int = 1

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        return [f"ivfflat.probes = {self.probes}"]

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return f"ivfflat.probes = {self.probes}"


@dataclass
class IVFIndex(BaseIndex):
    index_type: str = "ivf"
    lists: int = 100
    quantizer: str = field(
        default="sq8", init=False
    )  # Disable `quantizer` initialization currently only supports the value "sq8"

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        return f"(lists = {self.lists}, quantizer = {self.quantizer})"


@dataclass
class IVFQueryOptions(QueryOptions):
    probes: int = 1

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        return [f"ivf.probes = {self.probes}"]

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return f"ivf.probes = {self.probes}"


@dataclass
class ScaNNIndex(BaseIndex):
    """ScaNN index configuration for AlloyDB.

    See https://cloud.google.com/alloydb/docs/ai/create-scann-index.

    Args:
        num_leaves (int): Number of partitions. Used by the default and
            ``"MANUAL"`` modes. Defaults to 5.
        mode (Optional[str]): Keyword-only. ``"AUTO"`` lets AlloyDB choose the
            index parameters, so ``num_leaves`` and ``quantizer`` are not sent.
            ``"MANUAL"`` sends ``num_leaves`` and ``quantizer`` explicitly.
            Case-insensitive. Defaults to ``None``, which emits the same index
            options as previous releases.

    Raises:
        ValueError: If ``mode`` is not ``"AUTO"`` or ``"MANUAL"``.
    """

    index_type: str = "ScaNN"
    num_leaves: int = 5
    quantizer: str = field(
        default="sq8", init=False
    )  # Disable `quantizer` initialization currently only supports the value "sq8"
    # kw_only keeps the positional order of the existing fields unchanged.
    mode: Optional[str] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.mode is not None:
            if self.mode.upper() not in _SCANN_MODES:
                raise ValueError(
                    f"Invalid mode {self.mode!r}. Supported modes are 'AUTO' and 'MANUAL'."
                )
            self.mode = self.mode.upper()

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        if self.mode == "AUTO":
            return "(mode = 'AUTO')"
        if self.mode == "MANUAL":
            return f"(mode = 'MANUAL', num_leaves = {self.num_leaves}, quantizer = {self.quantizer})"
        return f"(num_leaves = {self.num_leaves}, quantizer = {self.quantizer})"


@dataclass
class ScaNNQueryOptions(QueryOptions):
    """Query options for a ScaNN index.

    See https://cloud.google.com/alloydb/docs/reference/ai/scann-index-reference.

    Args:
        num_leaves_to_search (int): Number of partitions to search. Defaults to 1.
        pre_reordering_num_neighbors (int): Number of candidate neighbors to
            consider before reordering. ``-1`` uses the server default.
            Defaults to -1.
        pct_leaves_to_search (Optional[float]): Percentage of partitions to
            search, between 0 and 100. When set, it is sent in addition to
            ``num_leaves_to_search`` and AlloyDB uses it to pick the number of
            partitions. Defaults to ``None``, which does not send it.

    Raises:
        ValueError: If ``pct_leaves_to_search`` is outside 0 to 100.
    """

    num_leaves_to_search: int = 1
    pre_reordering_num_neighbors: int = -1
    pct_leaves_to_search: Optional[float] = None

    def __post_init__(self) -> None:
        if self.pct_leaves_to_search is not None and not (
            _MIN_PCT_LEAVES_TO_SEARCH
            <= self.pct_leaves_to_search
            <= _MAX_PCT_LEAVES_TO_SEARCH
        ):
            raise ValueError(
                f"pct_leaves_to_search must be between {_MIN_PCT_LEAVES_TO_SEARCH} "
                f"and {_MAX_PCT_LEAVES_TO_SEARCH}."
            )

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        params = [
            f"scann.num_leaves_to_search = {self.num_leaves_to_search}",
            f"scann.pre_reordering_num_neighbors = {self.pre_reordering_num_neighbors}",
        ]
        if self.pct_leaves_to_search is not None:
            params.append(f"scann.pct_leaves_to_search = {self.pct_leaves_to_search}")
        return params

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return ", ".join(self.to_parameter())
