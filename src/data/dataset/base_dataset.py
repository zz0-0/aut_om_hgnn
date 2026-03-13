"""Base dataset class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from torch_geometric.data import Dataset, HeteroData  # type: ignore

from src.config.train_config import RobotType
from src.graph.morphology.base_morphology import RobotMorphology
from src.graph.spec.base_spec import BaseSpec


class BaseDataset(ABC, Dataset):
    """
    Abstract dataset that loads and assembles graph samples.

    RESPONSIBILITY:
    - Load Numpy memmap trajectory data
    - For each timestep:
        - Extract node features using FeatureExtractor
        - Assemble edge indices from Morphology
        - Assemble labels (contact states, GRF, COM)
    - Return HeteroData (x_dict, edge_index_dict, y_*)
    - Handle caching of processed samples

    DOES NOT:
    - Extract features (delegates to FeatureExtractor)
    - Parse robots (Parser's job)
    - Extract edge indices (Morphology's job)

    INPUT REQUIREMENTS:
    - Numpy memmap file path (trajectory data)
    - Morphology (edge structure, from Parser)
    - Spec (node types, feature requirements)
    - FeatureExtractor (spec-specific extraction logic)

    OUTPUT:
    - PyTorch Dataset that yields HeteroData samples
    """

    _registry: dict[RobotType, Self] = {}

    def __init__(self, root: str = "."):
        """Initialize dataset."""
        super().__init__(root)  # type: ignore

    @classmethod
    def register(cls, robot_type: RobotType):
        """
        Decorator to register a dataset implementation.
        """

        def decorator(dataset_cls: Self) -> Self:
            cls._registry[robot_type] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def create_dataset(
        cls,
        dataset_path: Path,
        morphology: RobotMorphology,
        spec: BaseSpec,
        robot_type: RobotType,
        history_length: int = 1,
    ) -> Self:
        """
        Factory method to create a dataset by robot type.

        INPUT:
        - dataset_path: Path to Numpy memmap trajectory file
        - morphology: Robot structure (from Parser)
        - spec: Graph specification
        - robot_type: What robot dataset is for (G1_29DOF, GO2, etc.)

        OUTPUT:
        - Instantiated dataset ready to use as PyTorch Dataset

        RAISES:
        - ValueError: If robot_type not registered
        """
        if robot_type not in cls._registry:
            raise ValueError(
                f"Unknown dataset type: {robot_type}. "
                f"Available: {list(cls._registry.keys())}"
            )

        dataset_cls: Self = cls._registry[robot_type]
        return dataset_cls.build_from(
            dataset_path=dataset_path,
            morphology=morphology,
            spec=spec,
            history_length=history_length,
        )

    @classmethod
    @abstractmethod
    def build_from(
        cls,
        dataset_path: Path,
        morphology: RobotMorphology,
        spec: BaseSpec,
        history_length: int = 1,
    ) -> Self:
        """
        Factory constructor. REQUIRED - each child implements this.

        This is the ONLY way to instantiate via factory pattern.
        Child classes must override and implement their initialization logic.

        INPUT:
        - dataset_path: Path to Numpy memmap trajectory file
        - morphology: Robot structure from parser
        - spec: Graph specification

        OUTPUT:
        - Initialized dataset ready to use as PyTorch Dataset
        """
        pass

    @abstractmethod
    def len(self) -> int:
        """
        Return total number of samples.

        OUTPUT:
        - int: number of HeteroData samples in dataset
        """
        pass

    @abstractmethod
    def get(self, idx: int) -> HeteroData:
        """
        Get a single sample.

        INPUT:
        - idx: Sample index

        OUTPUT:
        - HeteroData with:
            - x_dict: {node_type: feature_tensor}
            - edge_index_dict: {edge_tuple: edge_indices}
            - y_contact_states, y_contact_forces, y_com: labels
        """
        pass

    @abstractmethod
    def process(self):
        """
        Process raw Numpy memmap file into samples.

        RESPONSIBILITY:
        - Load Numpy memmap file
        - For each timestep:
            - Get raw data for the timestep
            - Use FeatureExtractor.extract() to get node features
            - Combine with Morphology's edge indices
            - Assemble labels
            - Create HeteroData sample
        - Cache processed samples (save to disk)

        This method is called once on first load.
        Subsequent loads use cached .pt file.
        """
        pass
