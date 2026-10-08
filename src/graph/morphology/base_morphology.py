"""Robot morphology data holder."""

from dataclasses import dataclass


from src.config.batch_schema import edge_index_dict_type
from src.graph.spec.base_spec import symmetry_permutation_dict_type


@dataclass
class RobotMorphology:
    """Data class to hold robot morphology information.
    The reason why we don't have x_dict is because the x_dict contains the real value from dataset.
    The morphology only contains the structure of the graph, which is defined by the node types and edge types.
    Here we define the node types and their mapping to the USD node paths, as well as the edge connectivity between the nodes.
    Later on, when processing the dataset, we will use the morphology to construct the graph and fill in the x_dict with the real node features from the dataset.
    """

    node_type_usd_node_dict: dict[str, list[str]]
    """Mapping from node type to list of USD node name."""

    node_type_usd_node_index_dict: dict[str, list[int]]
    """Mapping from node type to list of USD node indices (for graph construction)."""

    edge_index_dict: edge_index_dict_type
    """Edge connectivity: {(src_type, edge_type, dst_type): (2, num_edges)}"""

    symmetry_permutation_dict: symmetry_permutation_dict_type | None = None
    """Name-resolved symmetry row permutations for this robot instance.

    Rows are permuted with ``new_row[i] = old_row[permutation[i]]``. A permutation
    length equal to the number of rows encodes a full per-node permutation; shorter
    lengths are interpreted as contiguous row-group permutations.
    """
