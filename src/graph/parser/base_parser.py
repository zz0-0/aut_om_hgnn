"""Base parser class - extracts robot structure from description files."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from src.config.train_enum import ModelType
from src.config.train_config import RobotType
from src.graph.spec.base_spec import BaseSpec
from src.graph.morphology.base_morphology import RobotMorphology


class BaseParser(ABC):
    """
    Abstract parser that extracts robot structure from files.

    RESPONSIBILITY:
    - Parse robot description files (USD, URDF, etc.)
    - Extract joint hierarchy and body links
    - Return morphology with edge connectivity

    DOES NOT:
    - Load trajectory data
    - Extract features
    - Build datasets

    INPUT REQUIREMENTS:
    - File path (USD, URDF, etc.)
    - Spec (defines what structure is expected)

    OUTPUT:
    - RobotMorphology with edge indices and attributes
    """

    _registry: dict[tuple[str, RobotType], Self] = {}

    def __init__(self):
        """Initialize parser."""
        super().__init__()

    @classmethod
    def register(cls, suffix: str, robot_type: RobotType):
        """
        Decorator to register a parser for a file type + robot combo.
        """

        def decorator(parser_cls: Self) -> Self:
            key = (suffix, robot_type)
            cls._registry[key] = parser_cls
            return parser_cls

        return decorator

    @classmethod
    def create_parser(
        cls,
        robot_type: RobotType,
        model_type: ModelType,
        spec: BaseSpec,
        parser_path: Path,
    ) -> Self:
        """
        Factory method to create a parser by file type and robot type.

        INPUT:
        - robot_type: What robot we're parsing (G1_29DOF, GO2, etc.)
        - spec: Graph spec (validates structure)
        - parser_path: Path to robot description file

        OUTPUT:
        - Instantiated parser ready to call .parse()

        RAISES:
        - ValueError: If no parser registered for this combination
        """
        suffix = parser_path.suffix
        key = (suffix, robot_type)

        if key not in cls._registry:
            raise ValueError(
                f"No parser registered for {suffix} + {robot_type}. "
                f"Available: {list(cls._registry.keys())}"
            )

        parser_cls: Self = cls._registry[key]
        return parser_cls.build_from(model_type, spec, parser_path)

    @classmethod
    @abstractmethod
    def build_from(
        cls, model_type: ModelType, spec: BaseSpec, parser_path: Path
    ) -> Self:
        """
        Factory constructor. REQUIRED - each child implements this.

        This is the ONLY way to instantiate via factory pattern.
        Child classes must override and implement their initialization logic.

        INPUT:
        - model_type: Model type (e.g., MS_HGNN)
        - spec: Graph specification
        - parser_path: Path to robot description file

        OUTPUT:
        - Initialized parser instance ready to call parse()
        """
        pass

    @abstractmethod
    def find_physics_usd_path(self) -> Path:
        """
        Find the physics USD file path.

        RESPONSIBILITY:
        - Locate the USD file that contains the robot's physical structure
        - This may involve searching a directory, parsing a manifest, etc.

        OUTPUT:
        - Path to the physics USD file

        RAISES:
        - FileNotFoundError: If no valid USD file is found
        """
        pass

    @abstractmethod
    def parse(self) -> RobotMorphology:
        """
        Parse robot structure from file.

        RESPONSIBILITY:
        - Read the file (USD, URDF, etc.)
        - Extract joint hierarchy and links
        - Build edge connectivity
        - Validate against Spec

        OUTPUT:
        - RobotMorphology containing:
            - edge_indices: Dict[edge_type, edge_tensor]
            - edge_attrs: Dict[edge_type, attribute_tensor]
            - node_metadata: Names, positions, etc. (optional)

        RAISES:
        - FileNotFoundError: If file doesn't exist
        - ValueError: If structure doesn't match Spec
        """
        pass
