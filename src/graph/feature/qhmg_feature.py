"""QHMG feature extractor for quadruped robots."""

from typing import Dict, Any, Self, cast
import torch

from src.config.train_enum import SpecType
from src.graph.morphology.base_morphology import RobotMorphology
from src.graph.spec.qhmg import QHMGNodeType
from src.graph.feature.base_feature import BaseFeature, BaseFeatureType
from src.graph.spec.base_spec import BaseSpec


class QHMGFeatureExtractor(BaseFeature):
    """
    Feature extractor for QHMG (Quadruped Heterogeneous Morphological Graph).

    Extracts features for quadruped robots (e.g., Unitree GO2) according to QHMG specification.

    Node types and their input features:
    - base: inertial readings only (6D: lin_acc + ang_vel)
    - joint: compact motor features (3D: j_p, j_v, j_T)
    - foot: kinematic features (6D: f_p + f_v)
    """

    def __init__(self, spec: BaseSpec, morphology: RobotMorphology):
        """Initialize QHMG feature extractor."""
        super().__init__(spec)
        self.morphology = morphology
        self.node_type_names = {
            node_type: list(self.morphology.node_type_usd_node_dict.get(node_type, []))
            for node_type in (
                QHMGNodeType.BASE.value,
                QHMGNodeType.JOINT.value,
                QHMGNodeType.FOOT.value,
            )
        }

    @classmethod
    def build_from(cls, spec: BaseSpec, morphology: RobotMorphology) -> Self:
        """Factory constructor for QHMG feature extractor."""
        return cls(spec, morphology)

    @staticmethod
    def _to_vector3(value: torch.Tensor) -> torch.Tensor:
        tensor = value.to(torch.float32)
        if tensor.ndim > 1:
            tensor = tensor[0]
        tensor = tensor.reshape(-1)
        if tensor.numel() < 3:
            raise ValueError(
                f"Expected at least 3 values for IMU vector, got shape {tuple(value.shape)}"
            )
        return tensor[:3]

    def _joint_indices_for_type(
        self, raw_data: Dict[str, Any], node_type: str
    ) -> list[int]:
        joint_names = raw_data.get("joint_names")
        if not isinstance(joint_names, list):
            raise ValueError(
                "raw_data must include 'joint_names' list for feature mapping"
            )

        joint_name_items = cast(list[object], joint_names)
        joint_names_str = [str(name) for name in joint_name_items]
        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names_str)}

        def resolve_index(node_name: str) -> int:
            if node_name in joint_name_to_idx:
                return joint_name_to_idx[node_name]

            # GO2-style quadruped datasets often store distal actuator joints as
            # *_calf_joint while parser foot nodes can be named *_foot_joint.
            if node_type == QHMGNodeType.FOOT.value and node_name.endswith(
                "_foot_joint"
            ):
                mapped_name = node_name.replace("_foot_joint", "_calf_joint")
                if mapped_name in joint_name_to_idx:
                    return joint_name_to_idx[mapped_name]

            return -1

        indices: list[int] = []
        for name in self.node_type_names.get(node_type, []):
            indices.append(resolve_index(name))
        return indices

    def _build_motor_scalar_features(
        self,
        raw_data: Dict[str, Any],
        indices: list[int],
    ) -> torch.Tensor:
        if len(indices) == 0:
            return torch.zeros((0, 3), dtype=torch.float32)

        joint_pos = raw_data["joint_pos"].to(torch.float32)
        joint_vel = raw_data["joint_vel"].to(torch.float32)
        joint_torque = raw_data["joint_torque"].to(torch.float32)

        features = torch.zeros((len(indices), 3), dtype=torch.float32)
        valid_positions = [position for position, idx in enumerate(indices) if idx >= 0]
        if len(valid_positions) == 0:
            return features

        valid_indices = [indices[position] for position in valid_positions]
        idx_tensor = torch.tensor(valid_indices, dtype=torch.long)
        pos = joint_pos.index_select(0, idx_tensor).unsqueeze(1)
        vel = joint_vel.index_select(0, idx_tensor).unsqueeze(1)
        torque = joint_torque.index_select(0, idx_tensor).unsqueeze(1)

        valid_features = torch.cat([pos, vel, torque], dim=1)
        position_tensor = torch.tensor(valid_positions, dtype=torch.long)
        features[position_tensor] = valid_features
        return features

    def _extract_foot_kinematics(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        foot_pos = raw_data["foot_pos_w"].to(torch.float32)
        foot_vel = raw_data["foot_lin_vel_w"].to(torch.float32)

        if foot_pos.ndim != 2 or foot_vel.ndim != 2:
            raise ValueError(
                "Foot position and velocity inputs must be rank-2 tensors "
                f"but got {tuple(foot_pos.shape)} and {tuple(foot_vel.shape)}"
            )
        if foot_pos.shape[1] != 3 or foot_vel.shape[1] != 3:
            raise ValueError(
                "Foot position and velocity inputs must have last dim=3 "
                f"but got {tuple(foot_pos.shape)} and {tuple(foot_vel.shape)}"
            )

        expected = len(self.node_type_names[QHMGNodeType.FOOT.value])
        available = min(int(foot_pos.shape[0]), int(foot_vel.shape[0]))

        if expected <= 0:
            return torch.zeros((0, 6), dtype=torch.float32)

        out = torch.zeros((expected, 6), dtype=torch.float32)
        if available > 0:
            copy_count = min(expected, available)
            out[:copy_count, :3] = foot_pos[:copy_count, :]
            out[:copy_count, 3:] = foot_vel[:copy_count, :]
        return out

    def feature_type_layout(
        self,
    ) -> dict[str, list[tuple[int, int, BaseFeatureType]]]:
        return {
            QHMGNodeType.BASE.value: [
                (0, 3, BaseFeatureType.VECTOR_3D),
                (3, 6, BaseFeatureType.PSEUDOVECTOR_3D),
            ],
            QHMGNodeType.JOINT.value: [
                (0, 3, BaseFeatureType.SCALAR),
            ],
            QHMGNodeType.FOOT.value: [
                (0, 3, BaseFeatureType.VECTOR_3D),
                (3, 6, BaseFeatureType.VECTOR_3D),
            ],
        }

    def extract(self, raw_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract node features from raw timestep data for QHMG.

        INPUT:
        - raw_data: Dict containing raw sensor data for one timestep

        OUTPUT:
        - Dict[node_type, torch.Tensor]: features for each node type
        """
        # Extract base features (IMU only)
        base_features = self._extract_base_features(raw_data)

        # Extract joint features (compact motor scalars)
        joint_features = self._extract_joint_features(raw_data)

        # Extract foot features (kinematics)
        foot_features = self._extract_foot_features(raw_data)

        return {
            QHMGNodeType.BASE.value: base_features,
            QHMGNodeType.JOINT.value: joint_features,
            QHMGNodeType.FOOT.value: foot_features,
        }

    def _extract_base_features(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract base node features: IMU readings only.

        Features: [imu_lin_acc_x, imu_lin_acc_y, imu_lin_acc_z,
                  imu_ang_vel_x, imu_ang_vel_y, imu_ang_vel_z]
        Shape: [1, 6] (single base node)
        """
        imu_lin_acc = self._to_vector3(raw_data["imu_lin_acc"])
        imu_ang_vel = self._to_vector3(raw_data["imu_ang_vel"])
        base = torch.cat([imu_lin_acc, imu_ang_vel], dim=0).unsqueeze(0)
        return base

    def _extract_joint_features(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract joint node features: compact motor scalars.

        Features per joint: [pos, vel, torque]
        Shape: [num_joints, 3]
        """
        indices = self._joint_indices_for_type(raw_data, QHMGNodeType.JOINT.value)
        return self._build_motor_scalar_features(raw_data, indices)

    def _extract_foot_features(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Extract foot node features: foot kinematics.

        Features per foot: [f_p(3), f_v(3)]
        Shape: [num_feet, 6]
        """
        return self._extract_foot_kinematics(raw_data)


BaseFeature.register(SpecType.QHMG)(QHMGFeatureExtractor)  # type: ignore[arg-type]
