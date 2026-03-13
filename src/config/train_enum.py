from enum import Enum


class RobotType(Enum):
    """Supported robot types."""

    UNITREE_G1_29DOF = "UNITREE_G1_29DOF"
    UNITREE_G1_23DOF = "UNITREE_G1_23DOF"
    UNITREE_GO2 = "UNITREE_GO2"


class SpecType(Enum):
    """Supported graph specifications."""

    BHMG = "BHMG"  # Biped Heterogeneous Morphological Graph
    QHMG = "QHMG"  # Quadruped Heterogeneous Morphological Graph


class ModelType(Enum):
    """Supported model architectures."""

    MI_HGNN = "MI_HGNN"
    MS_HGNN = "MS_HGNN"


class SymmetryType(Enum):
    """Supported symmetry types."""

    C2 = "C2"  # left-right symmetry
    K4 = "K4"  # front-back + left-right symmetry


class Stage(Enum):
    """Training stage enum for tracking train/val/test phases."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class OutputType(Enum):
    """
    Output types to estimate.

    Each output type carries ALL its metadata:
    - What we're estimating (contact, GRF, COM)
    - Loss function for training
    - Output dimensions
    - Which node type produces this prediction

    This is the single source of truth for output specifications.
    """

    CONTACT = "CONTACT"
    GROUND_REACTION_FORCE = "GROUND_REACTION_FORCE"
    CENTER_OF_MASS = "CENTER_OF_MASS"
