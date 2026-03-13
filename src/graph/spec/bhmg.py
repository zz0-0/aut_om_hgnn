from typing import Self
from enum import Enum

from src.config.train_enum import OutputType, SpecType, SymmetryType
from src.graph.spec.base_spec import (
    BaseSpec,
    node_edge_relations_type,
    symmetry_edge_dict_type,
    symmetry_permutation_dict_type,
)


class BHMGNodeType(Enum):
    """BHMG node types - single source of truth for node definitions."""

    BASE = "base"  # Root body/torso
    JOINT = "joint"  # Motor/DOF
    FOOT = "foot"  # End-effector (in contact with ground)
    HAND = "hand"  # Upper limb End-effector (not in contact, but included for completeness)


class BHMGEdgeType(Enum):
    """BHMG edge types - single source of truth for edge definitions."""

    CONNECT = "connect"  # Parent-child relationships in kinematic tree


class BHMGSymmetryEdgeType(Enum):
    """BHMG symmetry edge types - single source of truth for symmetry edge definitions."""

    GS = "gs"  # biped only has left-right. Sagittal: left-right symmetry (L ↔ R)

    def flip_axes_index(self) -> list[int]:
        """Return the indices of the axes that should be flipped for this symmetry edge type."""
        if self == BHMGSymmetryEdgeType.GS:
            return [1]  # Flip y-axis for left-right symmetry
        else:
            raise ValueError(f"Unknown symmetry edge type: {self}")


class BHMG(BaseSpec):
    """
    Biped Heterogeneous Morphological Graph Specification.

    Defines the graph schema for biped robots (e.g., Unitree G1).

    Structure:
    - Nodes: base (torso), joint (motor), foot, hand
    - Edges: hierarchical connectivity from base through joints to feet/hands
    - Outputs: contact, GRF, COM predictions

    This spec is FIXED - it defines the topology for ALL biped robots in this system.
    Robot-specific variations (DOF count, morphology) come from the Parser/Morphology layer.
    """

    def __init__(self, symmetry_type: SymmetryType):
        """Initialize BHMG specification."""
        super().__init__()
        self.symmetry_type = symmetry_type
        self.symmetry_edge_dict = {
            SymmetryType.C2.value: {
                BHMGSymmetryEdgeType.GS.value: BHMGSymmetryEdgeType.GS.flip_axes_index(),
            },
        }

    @classmethod
    def build_from(cls, symmetry_type: SymmetryType) -> Self:
        """
        Factory constructor for BHMG.

        BHMG is a singleton spec with no parameters - all config is hardcoded.
        This factory method enables consistency with the factory pattern used
        by BaseSpec.create_spec().

        OUTPUT:
        - Initialized BHMG specification
        """
        return cls(symmetry_type)

    def node_types_with_history(self, history_length: int) -> dict[str, int]:
        if history_length < 1:
            raise ValueError("history_length must be >= 1")

        base_dims = {
            BHMGNodeType.BASE.value: 6,  # base linear acceleration (3) + base angular velocity (3)
            BHMGNodeType.JOINT.value: 3,  # joint position (1) + joint velocity (1) + joint torque (1)
            BHMGNodeType.FOOT.value: 6,  # foot position xyz (3) + foot linear velocity xyz (3)
            BHMGNodeType.HAND.value: 6,  # hand position xyz (3) + hand linear velocity xyz (3)
        }
        if history_length == 1:
            return base_dims
        return {node_type: dim * history_length for node_type, dim in base_dims.items()}

    def node_edge_relations(self) -> node_edge_relations_type:
        """
        Define edge types and connectivity patterns.

        BHMG uses a single edge type "connect" representing parent-child relationships
        in the kinematic tree.

        CONNECTIVITY:
        - base ↔ joint: root connects to all top-level joints (hips, shoulders)
        - joint ↔ joint: sequential motors in each limb (hip→knee→ankle, shoulder→elbow→wrist)
        - joint ↔ foot: distal joints connect to feet (ankle→foot)
        - foot ↔ joint: reverse edges for bidirectional message passing
        - joint ↔ hand: distal joints connect to hands (wrist→hand)
        - hand ↔ joint: reverse edges for bidirectional message passing

        EDGE DIRECTION:
        - Forward: parent → child (e.g., hip → knee)
        - Backward: child → parent (e.g., knee → hip)
        Both directions enable bidirectional GNN message passing.

        OUTPUT:
        - Dict[str, List[Tuple[str, str]]]:
            key: edge type name ("connect")
            value: list of (source_node_type, target_node_type) tuples
        """
        return {
            BHMGEdgeType.CONNECT.value: [
                (
                    BHMGNodeType.BASE.value,
                    BHMGNodeType.JOINT.value,
                ),  # Base to first joint
                (
                    BHMGNodeType.JOINT.value,
                    BHMGNodeType.BASE.value,
                ),  # First joint back to base
                (
                    BHMGNodeType.JOINT.value,
                    BHMGNodeType.JOINT.value,
                ),  # Sequential joints in chain
                (
                    BHMGNodeType.JOINT.value,
                    BHMGNodeType.FOOT.value,
                ),  # Distal joint to foot
                (
                    BHMGNodeType.FOOT.value,
                    BHMGNodeType.JOINT.value,
                ),  # Foot back to distal joint
                (
                    BHMGNodeType.JOINT.value,
                    BHMGNodeType.HAND.value,
                ),  # Distal joint to hand
                (
                    BHMGNodeType.HAND.value,
                    BHMGNodeType.JOINT.value,
                ),  # Hand back to distal joint
            ],
        }

    def node_edge_symmetry_relations(self) -> node_edge_relations_type:
        """
        Define symmetry edge types and symmetric node pairs.

        BHMG uses a single symmetry edge type "gs" representing left-right symmetry in bipeds.

        SYMMETRY:
        - Left-right symmetry between corresponding nodes on left and right sides of the body.
        - For example, left hip joint is symmetric to right hip joint, left foot is symmetric to right foot.

        OUTPUT:
        - Dict[str, List[Tuple[str, str]]]:
            key: symmetry edge type name ("gs")
            value: list of (node_type_1, node_type_2) tuples representing symmetric pairs
        """
        node_edge_symmetry_dict: node_edge_relations_type = {}

        symmetric_pairs = [
            (BHMGNodeType.BASE.value, BHMGNodeType.BASE.value),
            (BHMGNodeType.JOINT.value, BHMGNodeType.JOINT.value),
            (BHMGNodeType.FOOT.value, BHMGNodeType.FOOT.value),
            (BHMGNodeType.HAND.value, BHMGNodeType.HAND.value),
        ]

        for edge_type in self.symmetry_edge_dict[self.symmetry_type.value]:
            node_edge_symmetry_dict[edge_type] = symmetric_pairs

        return node_edge_symmetry_dict

    def output_node_type(self, output_type: OutputType) -> str:
        """
        Return which node type produces predictions for this output.

        OUTPUT:
        - Node type name: "foot", "base", etc.
        """
        if output_type == OutputType.CONTACT:
            return BHMGNodeType.FOOT.value
        elif output_type == OutputType.GROUND_REACTION_FORCE:
            return BHMGNodeType.FOOT.value
        elif output_type == OutputType.CENTER_OF_MASS:
            return BHMGNodeType.BASE.value
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    def symmetry_edge_mapping(self) -> symmetry_edge_dict_type:
        """
        Define which symmetry edge types correspond to each symmetry type.

        This is used to determine which edges in the graph represent the symmetries
        for a given Spec. For example, a C2 symmetry might have one edge type "gt"
        representing left-right symmetry, while a K4 symmetry might have multiple edge
        types representing different symmetry relations.

        OUTPUT:
            - Dict mapping:
                - key: SymmetryType (e.g., SymmetryType.C2, SymmetryType.K4)
                - value: list of edge type names (e.g., ["gt", "gs"]) that represent the symmetries for that type
        """
        return self.symmetry_edge_dict

    def symmetry_permutation_mapping(self) -> symmetry_permutation_dict_type:
        """
        Define group-level row permutations for BHMG symmetry operators.

        BHMG is modeled as 2 symmetric limb groups (left/right) for non-base nodes.
        """
        return {
            SymmetryType.C2.value: {
                BHMGNodeType.JOINT.value: {
                    BHMGSymmetryEdgeType.GS.value: [1, 0],
                },
                BHMGNodeType.FOOT.value: {
                    BHMGSymmetryEdgeType.GS.value: [1, 0],
                },
                BHMGNodeType.HAND.value: {
                    BHMGSymmetryEdgeType.GS.value: [1, 0],
                },
            }
        }


BaseSpec.register(SpecType.BHMG)(BHMG)  # type: ignore[arg-type]
