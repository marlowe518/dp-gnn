"""Subgraph extraction interface. Stub only.

Will build subgraph (nodes + edges) for a minibatch from sampling output.
Target: reference repo subgraph extraction. Paper: subgraph definition in algorithm.
"""

from __future__ import annotations

from typing import Any


def extract_subgraph(sampled_nodes: Any, edge_index: Any, **kwargs: Any) -> Any:
    """Extract subgraph for batch. TODO: implement; return subgraph (e.g. PyG Data).

    Target: reference repo subgraph module. Paper: same subgraph as in Alg.
    """
    # TODO: implement extraction; no implementation in skeleton
    raise NotImplementedError("Subgraph extraction not implemented yet (skeleton only)")
