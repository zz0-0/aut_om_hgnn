from typing import Dict, Self

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GraphConv, HeteroDictLinear  # type: ignore
from src.config.train_enum import ModelType, OutputType
from src.config.train_config import TrainConfig
from src.config.batch_schema import x_dict_type, edge_index_dict_type
from src.model.architecture.base_model import BaseModel
from src.graph.spec.base_spec import BaseSpec


class MI_HGNN(BaseModel):
    """
    Morphology-informed Heterogeneous Graph Neural Network.

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
    """

    def __init__(
        self,
        train_config: TrainConfig,
        spec: BaseSpec,
    ):
        """
        Initialize MI_HGNN.

        Args:
            train_config: Training configuration
            spec: Graph specification defining node types, edges, output
        """
        super().__init__()

        # Store config and spec for reference
        self.train_config = train_config
        self.spec = spec

        # Extract configuration
        self.hidden_channels = train_config.hidden_channels
        self.num_layers = train_config.num_layers
        self.activation = train_config.activation
        self.output_type = train_config.output_type

        # Get node and edge information from spec
        node_types_dict = self.spec.node_types_with_history(
            self.train_config.history_length
        )  # Dict[str, int] - input dimensions
        node_edge_rels = (
            self.spec.node_edge_relations()
        )  # Dict[str, List[Tuple[str, str]]] - edge relations

        # ===== ENCODER =====
        # Project heterogeneous node features to common hidden dimension
        self.encoder = HeteroDictLinear(
            in_channels=node_types_dict,  # {node_type: input_dim, ...}
            out_channels=self.hidden_channels,  # All project to same hidden dim, to make sure they can be processed by same conv layers
        )

        # ===== GRAPH CONVOLUTION LAYERS =====
        # For each layer, create HeteroConv with all edge types
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            conv_dict = {}

            # MI-HGNN ONLY processes "connect" edges (kinematic connectivity)
            # Ignores symmetry edges (gt, gs) - those are MS-HGNN only
            for edge_type, node_pairs in node_edge_rels.items():
                if edge_type == "connect":
                    for source_node_type, target_node_type in node_pairs:
                        # Key format: (source_type, edge_type, target_type)
                        conv_key = (source_node_type, edge_type, target_node_type)

                        # Create GraphConv for this edge type
                        # MI-HGNN uses "add" aggregation for all kinematic edges
                        conv_dict[conv_key] = GraphConv(
                            in_channels=self.hidden_channels,
                            out_channels=self.hidden_channels,
                            aggr="add",  # Aggregate messages with addition
                        )

            # Combine all edge convolutions with HeteroConv
            hetero_conv = HeteroConv(conv_dict, aggr="sum")  # type: ignore
            self.convs.append(hetero_conv)

        # ===== DECODER =====
        # Determine which node type to decode from (based on output_type)
        # Each output type knows which node type should produce predictions
        self.output_node_type = self.spec.output_node_type(self.output_type)

        # Each output type knows its output dimension
        output_dim = self.spec.output_channels(self.output_type)
        self.decoder = nn.Linear(self.hidden_channels, output_dim)

    @classmethod
    def build_from(
        cls,
        train_config: TrainConfig,
        spec: BaseSpec,
    ) -> Self:
        """
        Factory constructor for MI_HGNN.

        Args:
            train_config: Training configuration
            spec: Graph specification

        Returns:
            Initialized MI_HGNN model
        """
        return cls(train_config, spec)

    def forward(
        self,
        x_dict: x_dict_type,
        edge_index_dict: edge_index_dict_type,
    ) -> Dict[OutputType, torch.Tensor]:
        """
        Forward pass through MI_HGNN.

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

        # ===== ENCODER: Project to hidden dimension =====
        x_dict = self.encoder(x_dict)

        # Apply activation to encoded features
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}

        # ===== GRAPH CONVOLUTION: Message passing =====
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

            # Apply activation between layers
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}

        # ===== DECODER: Project to output =====
        # Extract features for the output node type (e.g., "foot" for CONTACT)
        output_node_features = x_dict[self.output_node_type]

        # Decode to output dimension
        out = self.decoder(output_node_features)  # [num_output_nodes, output_dim]

        # Return as dict with output type as key
        return {self.output_type: out}


BaseModel.register(ModelType.MI_HGNN)(MI_HGNN)  # type: ignore[arg-type]
