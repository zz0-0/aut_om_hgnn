"""Base feature extractor class."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Self

import torch

from src.config.train_enum import SpecType
from src.graph.morphology.base_morphology import RobotMorphology
from src.graph.spec.bhmg import BHMG
from src.graph.spec.qhmg import QHMG
from src.graph.spec.base_spec import BaseSpec


class BaseFeatureType(Enum):
    VECTOR_3D = "vector_3d"  # 3D vector (e.g., position, velocity)
    PSEUDOVECTOR_3D = "pseudovector_3d"  # 3D pseudovector (e.g., angular velocity, which changes sign under reflection)
    SCALAR = "scalar"  # Scalar value (e.g., contact state, which is binary and does not change under reflection)


class BaseFeature(ABC):
    """
    Abstract feature extractor that assembles node FEATURES (inputs only) from raw data.

    KEY PRINCIPLE: Extract INPUTS to the graph, NOT outputs/labels
    - Features are what the GNN sees (fed to graph neural network)
    - Labels are what we predict (output of graph neural network)
    - They should NOT heavily overlap to avoid circular dependencies

    RESPONSIBILITY:
    - Extract INPUT node features from raw Numpy memmap timestep data
    - Assemble features specific to each node type
    - Handle robot-specific logic (different DOF counts, IMU configurations)
    - Handle morphology-specific logic (BHMG ≠ QHMG extraction differs)
    - Return feature dict matching Spec's node types
    - Ensure no circular dependencies (predictions not fed as inputs)

    DOES NOT:
    - Extract labels/outputs (Dataset's job)
    - Load Numpy memmap files (Dataset's job)
    - Manage datasets
    - Create edge indices (Morphology's job)

    WHY SEPARATE MODULE:
    - Bloated code in dataset if we mix feature extraction logic
    - Needs to be testable: "Given raw data, produce correct INPUT features"
    - Different specs extract differently (BHMG vs QHMG)
    - Reusable across different scripts/tools
    - Enables robot-specific and spec-specific implementations

    INPUT REQUIREMENTS:
    - Spec: defines what node types and features are needed
    - Raw timestep data: ONLY the fields needed for graph inputs
      (Do not pass label fields like contact_states if they shouldn't be in features)

    OUTPUT:
    - Dict[node_type, torch.Tensor]: features for each node type
    """

    _registry: dict[SpecType, Self] = {}

    def __init__(self, spec: BaseSpec):
        """
        Initialize feature extractor.

        INPUT:
        - spec: Graph specification that defines node types and requirements
        """
        self.spec = spec

    @classmethod
    def register(cls, spec_type: SpecType):
        """
        Decorator to register a feature extractor implementation.
        """

        def decorator(extractor_cls: Self) -> Self:
            cls._registry[spec_type] = extractor_cls
            return extractor_cls

        return decorator

    @classmethod
    def create_extractor(cls, spec: BaseSpec, morphology: RobotMorphology) -> Self:
        """
        Factory method to create a feature extractor by spec type.

        INPUT:
        - spec_type: Type of spec (e.g., "BHMG", "QHMG")
        - spec: Graph specification (passed to extractor)
        - morphology: Robot morphology (used to determine node counts and types)

        OUTPUT:
        - Instantiated feature extractor

        RAISES:
        - ValueError: If spec_type not registered
        """
        if isinstance(spec, BHMG):
            spec_type = SpecType.BHMG
        elif isinstance(spec, QHMG):
            spec_type = SpecType.QHMG
        else:
            raise ValueError(f"Unknown spec type: {type(spec)}. Must be BHMG or QHMG.")

        if spec_type not in cls._registry:
            raise ValueError(
                f"Unknown spec type: {spec_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        extractor_cls: Self = cls._registry[spec_type]
        return extractor_cls.build_from(spec, morphology)

    @classmethod
    @abstractmethod
    def build_from(cls, spec: BaseSpec, morphology: RobotMorphology) -> Self:
        """
        Factory constructor. REQUIRED - each child implements this.

        This is the ONLY way to instantiate via factory pattern.
        Child classes must override and implement their initialization logic.

        INPUT:
        - spec: Graph specification
        - morphology: Robot morphology (used to determine node counts and types)
        OUTPUT:
        - Initialized feature extractor ready to call extract()
        """
        pass

    @abstractmethod
    def extract(self, raw_data: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Extract node features from raw timestep data.

        INPUT:
        - raw_data: Dict of raw arrays for one timestep
            Keys might include: "joint_pos", "joint_vel", "joint_acc", "root_com_pos",
                                "root_com_quat", "base_lin_vel", "base_ang_vel",
                                "imu_lin_acc", "imu_ang_vel", "contact_states", "contact_forces"
            Each value is a numpy array or tensor

        OUTPUT:
        - Dict mapping:
            - key: node type name (e.g., "base", "joint", "foot")
            - value: torch.Tensor of shape [num_nodes, input_channels]
                  where input_channels matches Spec.node_types()
        """
        pass

    def feature_type_layout(
        self,
    ) -> dict[str, list[tuple[int, int, BaseFeatureType]]]:
        """
        Optional semantic layout for node feature channels.

        OUTPUT:
        - Dict[node_type, List[(start_idx, end_idx, feature_type)]]
          where [start_idx:end_idx] is a contiguous feature slice.
        """
        return {}

    def feature_type_layout_serialized(
        self,
    ) -> dict[str, list[tuple[int, int, str]]]:
        """Serialize feature_type_layout using string values for storage in HeteroData."""
        return {
            node_type: [
                (start, end, feature_type.value) for start, end, feature_type in blocks
            ]
            for node_type, blocks in self.feature_type_layout().items()
        }
