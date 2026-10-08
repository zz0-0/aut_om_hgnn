"""Batch data schema and protocol definitions."""

from torch_geometric.data import HeteroData  # type: ignore
import torch

x_dict_type = dict[str, torch.Tensor]
edge_index_dict_type = dict[tuple[str, str, str], torch.Tensor]


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
    - y_contact: Ground truth per-foot contact state labels
    - y_ground_reaction_force: Ground truth per-foot GRF labels
    - y_base_velocity: Ground truth base linear/angular velocity labels
    - y_total_ground_reaction_force: Ground truth summed GRF labels
    - y_base_angular_acceleration: Ground truth body-frame base angular acceleration
    - y_joint_acceleration: Ground truth joint acceleration labels
    - y_joint_friction: Ground truth joint friction torque labels
    """

    x_dict: x_dict_type
    """Node features: {node_type: (num_nodes, feature_dim)}"""

    edge_index_dict: edge_index_dict_type
    """Edge connectivity: {(src_type, edge_type, dst_type): (2, num_edges)}"""

    y_contact: torch.Tensor
    """Contact state labels: (batch_size, num_feet)"""

    y_ground_reaction_force: torch.Tensor
    """Per-foot GRF labels: (batch_size, num_feet * 3)"""

    y_base_velocity: torch.Tensor
    """Base linear + angular velocity labels: (batch_size, 6)"""

    y_total_ground_reaction_force: torch.Tensor
    """Summed ground reaction force labels: (batch_size, 3)"""

    y_base_angular_acceleration: torch.Tensor
    """Body-frame base angular acceleration labels: (batch_size, 3)"""

    y_joint_acceleration: torch.Tensor
    """Joint acceleration labels: (batch_size, num_joints)"""

    y_joint_friction: torch.Tensor
    """Joint friction torque labels: (batch_size, num_joints)"""
