from typing import Self

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GraphConv, HeteroDictLinear  # type: ignore

from src.config.train_enum import ModelType, OutputType
from src.config.train_config import TrainConfig
from src.config.batch_schema import x_dict_type, edge_index_dict_type
from src.model.architecture.base_model import BaseModel
from src.graph.spec.base_spec import BaseSpec


class MS_HGNN(BaseModel):
    """
    Morphology-symmetry Heterogeneous Graph Neural Network.

    Architecture:
    1. Encoder: Project node features to hidden dimension
       - Input: Dict[node_type] → tensor per node type
       - Output: Dict[node_type] → hidden_channels per node

    2. Graph Conv Layers: Message passing between nodes
       - For each layer and edge type: GraphConv with aggregation
       - Apply activation between layers

    3. Decoder: Project output node features to prediction
       - Input: hidden_channels from selected node type
       - Output: output_channels for the predicted quantity

    Predicts a SINGLE output type (e.g., CONTACT OR GRF OR COM, not multiple).
    For multi-output, see MS_HGNN.
    """

    def __init__(
        self,
        train_config: TrainConfig,
        spec: BaseSpec,
    ):
        """
        Initialize MS_HGNN.

        Args:
            train_config: Training configuration
            spec: Graph specification defining node types, edges, output
            reflection_coefficients: Optional dict with keys reflection_Q_js, reflection_Q_fs,
                                   reflection_Q_bs_lin, reflection_Q_bs_ang from Parser.
                                   If provided, these are used to initialize symmetry transformations.
        """
        super().__init__()

        self.train_config = train_config
        self.spec = spec

        # Create symmetry operator (MS-HGNN specific, not in spec)
        # self.symmetry_operator = MorphologySymmetryOperator(spec)

        # # Load reflection coefficients if provided
        # if reflection_coefficients is not None:
        #     self.symmetry_operator.set_reflection_coefficients(
        #         **reflection_coefficients
        #     )

        # Extract configuration
        self.hidden_channels = train_config.hidden_channels
        self.num_layers = train_config.num_layers
        self.activation = train_config.activation
        self.output_type = train_config.output_type

        # Get node and edge information from spec
        node_types_dict = self.spec.node_types_with_history(
            self.train_config.history_length
        )  # Dict[str, int] - input dimensions
        node_edge_rels = self.spec.node_edge_relations()
        # Dict[str, List[Tuple[str, str]]]
        node_edge_symmetry_rels = self.spec.node_edge_symmetry_relations()
        # ===== ENCODER =====
        # Project heterogeneous node features to common hidden dimension
        self.encoder = HeteroDictLinear(
            in_channels=node_types_dict,  # {node_type: input_dim, ...}
            out_channels=self.hidden_channels,  # All project to same hidden dim
        )

        # ===== GRAPH CONVOLUTION LAYERS =====
        # For each layer, create HeteroConv with all edge types
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            conv_dict = {}

            # For each edge type, create GraphConv for all (source, target) pairs
            for edge_type, node_pairs in node_edge_rels.items():
                for source_node_type, target_node_type in node_pairs:
                    # Key format: (source_type, edge_type, target_type)
                    conv_key = (source_node_type, edge_type, target_node_type)
                    # Create GraphConv for this edge type
                    conv_dict[conv_key] = GraphConv(
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        aggr="add",  # Use appropriate aggregation based on edge type
                    )

            for edge_type, node_pairs in node_edge_symmetry_rels.items():
                for source_node_type, target_node_type in node_pairs:
                    conv_key = (source_node_type, edge_type, target_node_type)
                    conv_dict[conv_key] = GraphConv(
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        aggr="mean",  # Use appropriate aggregation based on edge type
                    )

            # Combine all edge convolutions with HeteroConv
            hetero_conv = HeteroConv(conv_dict, aggr="sum")  # type: ignore
            self.convs.append(hetero_conv)

        # ===== BASE TRANSFORM LAYER (MS-HGNN specific) =====
        # Extra processing for base nodes to learn complex body dynamics
        self.base_transform = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            self.activation,  # Use activation instance, don't call it
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

        # ===== DECODER =====
        # Determine which node type to decode from (based on output_type)
        # Each output type knows which node type should produce predictions
        self.output_node_type = self.spec.output_node_type(self.output_type)

        # Each output type knows its output dimension
        output_dim = self.spec.output_channels(self.output_type)
        self.decoder = nn.Linear(self.hidden_channels, output_dim)

    @classmethod
    def build_from(cls, train_config: TrainConfig, spec: BaseSpec) -> Self:
        """
        Factory constructor for MS_HGNN.

        Args:
            train_config: Training configuration
            spec: Graph specification

        Returns:
            Initialized MS_HGNN model
        """
        return cls(train_config, spec)

    def forward(
        self,
        x_dict: x_dict_type,
        edge_index_dict: edge_index_dict_type,
    ) -> dict[OutputType, torch.Tensor]:
        """
        Forward pass through MS_HGNN.

        Args:
            x_dict: Node features per type
                Keys: node type names (e.g., "base", "joint", "foot", etc.)
                Values: [num_nodes_of_type, input_channels]

            edge_index_dict: Edge connectivity per type
                Keys: (source_type, edge_type, target_type) tuples
                Values: [2, num_edges] - source and target node indices

        Returns:
            Dict with single entry {output_type: prediction_tensor}
                prediction_tensor: [num_output_nodes, output_channels]
                    - CONTACT predictions: [num_feet, 1]
                    - GRF predictions: [num_feet, 3]
                    - COM predictions: [1, 6]
        """
        # ===== APPLY SYMMETRY ======
        # x_dict = self.symmetry_operator.apply_symmetry(x_dict)

        # ===== ENCODER: Project to hidden dimension =====
        x_dict = self.encoder(x_dict)

        # Apply activation to encoded features
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}

        # ===== GRAPH CONVOLUTION: Message passing with residual connections =====
        for conv in self.convs:
            # Compute new features from message passing
            x_dict_new = conv(x_dict, edge_index_dict)

            # Apply base_transform to base nodes, activation to others
            x_dict_new = {
                key: (self.base_transform(x) if key == "base" else self.activation(x))
                for key, x in x_dict_new.items()
            }

            # Add residual connections: x_new = x_new + x_old
            x_dict = {
                key: (
                    x_dict_new[key] + x_dict[key]
                    if key in x_dict and x_dict[key].shape == x_dict_new[key].shape
                    else x_dict_new[key]
                )
                for key in x_dict_new
            }

        # ===== DECODER: Project to output =====
        # Extract features for the output node type (e.g., "foot" for CONTACT)
        output_node_features = x_dict[self.output_node_type]

        # Decode to output dimension
        out = self.decoder(output_node_features)  # [num_output_nodes, output_dim]

        # Return as dict with output type as key
        return {self.output_type: out}


BaseModel.register(ModelType.MS_HGNN)(MS_HGNN)  # type: ignore[arg-type]
