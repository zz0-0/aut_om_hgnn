"""Robot morphology data holder."""

from dataclasses import dataclass


from src.config.batch_schema import edge_index_dict_type


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
