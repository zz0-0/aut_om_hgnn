"""Batch data schema and protocol definitions."""

from torch_geometric.data import HeteroData  # type: ignore
import torch

type x_dict_type = dict[str, torch.Tensor]
type edge_index_dict_type = dict[tuple[str, str, str], torch.Tensor]


class HeteroDataBatch(HeteroData):
    """
    Protocol defining the structure of a batched HeteroData sample.

    This defines the contract between:
    - Dataset: Creates HeteroData samples with these fields
    - Model Training: Expects batches with this exact structure

    This enables IDE autocomplete and type checking across the pipeline.

    FIELDS:
    - x_dict: Node features for each node type
    - edge_index_dict: Edge connectivity for each edge type
    - y_contact_states: Ground truth contact state labels
    - y_contact_forces: Ground truth ground reaction force (GRF) labels
    - y_com: Ground truth center of mass labels
    """

    x_dict: x_dict_type
    """Node features: {node_type: (num_nodes, feature_dim)}"""

    edge_index_dict: edge_index_dict_type
    """Edge connectivity: {(src_type, edge_type, dst_type): (2, num_edges)}"""

    y_contact_states: torch.Tensor
    """Contact state labels: (batch_size, num_feet)"""

    y_contact_forces: torch.Tensor
    """Ground reaction force labels: (batch_size, num_feet, 3)"""

    y_com: torch.Tensor
    """Center of mass labels: (batch_size, 3)"""
