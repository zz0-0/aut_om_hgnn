"""Unitree GO2 dataset - loads Numpy memmap trajectory data and assembles heterogeneous graph samples."""

from pathlib import Path
from typing import Self
import logging

from src.config.train_enum import RobotType
from src.data.dataset.base_dataset import BaseDataset
from src.data.dataset.unitree_g1_dataset import UnitreeG1Dataset
from src.graph.morphology.base_morphology import RobotMorphology
from src.graph.spec.base_spec import BaseSpec


class UnitreeGO2Dataset(UnitreeG1Dataset):
    """
    Dataset for Unitree GO2 quadruped.

    Uses the same Numpy memmap field layout as G1, with quadruped morphology.
    """

    def __init__(
        self,
        dataset_path: Path,
        morphology: RobotMorphology,
        spec: BaseSpec,
        history_length: int = 1,
    ):
        super().__init__(
            dataset_path=dataset_path,
            morphology=morphology,
            spec=spec,
            history_length=history_length,
        )
        self.logger = logging.getLogger(__name__)

    @classmethod
    def build_from(
        cls,
        dataset_path: Path,
        morphology: RobotMorphology,
        spec: BaseSpec,
        history_length: int = 1,
    ) -> Self:
        return cls(
            dataset_path=dataset_path,
            morphology=morphology,
            spec=spec,
            history_length=history_length,
        )


BaseDataset.register(RobotType.UNITREE_GO2)(UnitreeGO2Dataset)  # type: ignore[arg-type]
