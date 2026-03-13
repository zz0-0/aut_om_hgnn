from enum import Enum
from typing import Self


from src.config.train_enum import OutputType, SpecType, SymmetryType
from src.graph.spec.base_spec import (
    BaseSpec,
    node_edge_relations_type,
    symmetry_edge_dict_type,
    symmetry_permutation_dict_type,
)


class QHMGNodeType(Enum):
    """QHMG node types - single source of truth for node definitions."""

    BASE = "base"  # Root body/torso
    JOINT = "joint"  # Motor/DOF
    FOOT = "foot"  # End-effector (in contact with ground)


class QHMGEdgeType(Enum):
    """QHMG edge types - single source of truth for edge definitions."""

    CONNECT = "connect"  # Parent-child relationships in kinematic tree


class QHMGSymmetryEdgeType(Enum):
    """QHMG symmetry edge types - single source of truth for symmetry edge definitions."""

    GT = "gt"  # quadruped front-back symmetry (F ↔ B), i.e. transversal reflection
    GS = "gs"  # quadruped left-right symmetry (L ↔ R), i.e. sagittal reflection
    GR = "gr"  # rotational symmetry (R ↔ R) - 180 degree rotation around vertical axis

    def flip_axes_index(self) -> list[int]:
        """Return the indices of the axes that should be flipped for this symmetry edge type."""
        if self == QHMGSymmetryEdgeType.GT:
            return [0]  # Flip x-axis for front-back symmetry
        elif self == QHMGSymmetryEdgeType.GS:
            return [1]  # Flip y-axis for left-right symmetry
        elif self == QHMGSymmetryEdgeType.GR:
            return [0, 1]  # Flip both x and y axes for rotational symmetry
        else:
            raise ValueError(f"Unknown symmetry edge type: {self}")


class QHMG(BaseSpec):
    """
    Quadruped Heterogeneous Morphological Graph Specification.

    Defines the graph schema for quadruped robots (e.g., Unitree Go2).

    Structure:
    - Nodes: base (torso), joint (motor), foot, hand
    - Edges: hierarchical connectivity from base through joints to feet/hands
    - Outputs: contact, GRF, COM predictions

    This spec is FIXED - it defines the topology for ALL quadruped robots in this system.
    Robot-specific variations (DOF count, morphology) come from the Parser/Morphology layer.
    """

    def __init__(self, symmetry_type: SymmetryType):
        """Initialize QHMG specification."""
        super().__init__()
        self.symmetry_type = symmetry_type
        self.symmetry_edge_dict = {
            SymmetryType.K4.value: {
                QHMGSymmetryEdgeType.GT.value: QHMGSymmetryEdgeType.GT.flip_axes_index(),
                QHMGSymmetryEdgeType.GS.value: QHMGSymmetryEdgeType.GS.flip_axes_index(),
            },
        }

    @classmethod
    def build_from(cls, symmetry_type: SymmetryType) -> Self:
        """
        Factory constructor for QHMG.

        QHMG is a singleton spec with no parameters - all config is hardcoded.
        This factory method enables consistency with the factory pattern used
        by BaseSpec.create_spec().

        OUTPUT:
        - Initialized QHMG specification
        """
        return cls(symmetry_type)

    def node_types_with_history(self, history_length: int) -> dict[str, int]:
        if history_length < 1:
            raise ValueError("history_length must be >= 1")

        base_dims = {
            QHMGNodeType.BASE.value: 6,  # lin_acc(3) + ang_vel(3)
            QHMGNodeType.JOINT.value: 3,  # j_p(1) + j_v(1) + j_T(1)
            QHMGNodeType.FOOT.value: 6,  # f_p(3) + f_v(3)
        }
        if history_length == 1:
            return base_dims
        return {node_type: dim * history_length for node_type, dim in base_dims.items()}

    def node_edge_relations(self) -> node_edge_relations_type:
        """
        Define edge types and connectivity patterns.

        QHMG uses a single edge type "connect" representing parent-child relationships
        in the kinematic tree.

        CONNECTIVITY:
        - base ↔ joint: root connects to all top-level joints (hips, shoulders)
        - joint ↔ joint: sequential motors in each limb (hip→knee→ankle, shoulder→elbow→wrist)
        - joint ↔ foot: distal joints connect to feet (ankle→foot)
        - foot ↔ joint: reverse edges for bidirectional message passing

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
            QHMGEdgeType.CONNECT.value: [
                (
                    QHMGNodeType.BASE.value,
                    QHMGNodeType.JOINT.value,
                ),  # Base to first joint
                (
                    QHMGNodeType.JOINT.value,
                    QHMGNodeType.BASE.value,
                ),  # First joint back to base
                (
                    QHMGNodeType.JOINT.value,
                    QHMGNodeType.JOINT.value,
                ),  # Sequential joints in chain
                (
                    QHMGNodeType.JOINT.value,
                    QHMGNodeType.FOOT.value,
                ),  # Distal joint to foot
                (
                    QHMGNodeType.FOOT.value,
                    QHMGNodeType.JOINT.value,
                ),  # Foot back to distal joint
            ],
        }

    def node_edge_symmetry_relations(self) -> node_edge_relations_type:
        """
        Define symmetry edge types and symmetric node pairs.

        QHMG uses symmetry edge types for supported settings:
        - "gt": front-back symmetry (F ↔ B)
        - "gs": left-right symmetry (L ↔ R)

        SYMMETRY:
        - Left-right symmetry between corresponding nodes on left and right sides of the body.
        - For example, left hip joint is symmetric to right hip joint, left foot is symmetric to right foot.
        - Front-back symmetry between corresponding nodes on front and back sides of the body.
        - For example, front-left hip joint is symmetric to back-right hip joint.
        OUTPUT:
        - Dict[str, List[Tuple[str, str]]]:
            key: symmetry edge type name ("gt", "gs")
            value: list of (node_type_1, node_type_2) tuples representing symmetric pairs
        """
        node_edge_symmetry_dict: node_edge_relations_type = {}

        symmetric_pairs = [
            (QHMGNodeType.BASE.value, QHMGNodeType.BASE.value),
            (QHMGNodeType.JOINT.value, QHMGNodeType.JOINT.value),
            (QHMGNodeType.FOOT.value, QHMGNodeType.FOOT.value),
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
            return QHMGNodeType.FOOT.value
        elif output_type == OutputType.GROUND_REACTION_FORCE:
            return QHMGNodeType.FOOT.value
        elif output_type == OutputType.CENTER_OF_MASS:
            return QHMGNodeType.BASE.value
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
        Define group-level row permutations for QHMG symmetry operators.

        QHMG is modeled as 4 symmetric limb groups for non-base nodes:
        [FL, FR, BL, BR] in canonical order.
        """
        operator_map = {
            QHMGSymmetryEdgeType.GT.value: [2, 3, 0, 1],
            QHMGSymmetryEdgeType.GS.value: [1, 0, 3, 2],
            QHMGSymmetryEdgeType.GR.value: [3, 2, 1, 0],
        }

        return {
            SymmetryType.K4.value: {
                QHMGNodeType.JOINT.value: {
                    QHMGSymmetryEdgeType.GT.value: operator_map[
                        QHMGSymmetryEdgeType.GT.value
                    ],
                    QHMGSymmetryEdgeType.GS.value: operator_map[
                        QHMGSymmetryEdgeType.GS.value
                    ],
                },
                QHMGNodeType.FOOT.value: {
                    QHMGSymmetryEdgeType.GT.value: operator_map[
                        QHMGSymmetryEdgeType.GT.value
                    ],
                    QHMGSymmetryEdgeType.GS.value: operator_map[
                        QHMGSymmetryEdgeType.GS.value
                    ],
                },
            },
        }


BaseSpec.register(SpecType.QHMG)(QHMG)  # type: ignore[arg-type]
