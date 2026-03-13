"""BHMG feature extractor for biped robots."""

from typing import Any, Self, cast
import torch

from src.config.train_enum import SpecType
from src.graph.spec.bhmg import BHMGNodeType
from src.graph.feature.base_feature import BaseFeature, BaseFeatureType
from src.graph.spec.base_spec import BaseSpec
from src.graph.morphology.base_morphology import RobotMorphology


class BHMGFeatureExtractor(BaseFeature):
    """
    Feature extractor for BHMG (Biped Heterogeneous Morphological Graph).

    Extracts features for biped robots (e.g., Unitree G1) according to BHMG specification.

    Node types and their input features:
        - base: inertial readings
            Channels: [base linear acceleration xyz, base angular velocity xyz] (6D)
        - joint: motor readings
            Channels: [joint position, joint velocity, joint torque] (3D)
        - foot: foot kinematics
            Channels: [foot position xyz, foot linear velocity xyz] (6D)
        - hand: hand kinematics
            Channels: [hand position xyz, hand linear velocity xyz] (6D)
    """

    def __init__(self, spec: BaseSpec, morphology: RobotMorphology):
        """
        Initialize BHMG feature extractor.

        Args:
            spec: Graph specification
            morphology: Robot morphology with node_metadata for joint classification.
                       Required to correctly partition joint data into joint, foot, and hand
                       nodes based on parser's classification, and to determine node counts.
        """
        super().__init__(spec)
        self.morphology = morphology
        self.node_type_names = {
            node_type: list(self.morphology.node_type_usd_node_dict.get(node_type, []))
            for node_type in (
                BHMGNodeType.BASE.value,
                BHMGNodeType.JOINT.value,
                BHMGNodeType.FOOT.value,
                BHMGNodeType.HAND.value,
            )
        }
        self.base_count = max(1, len(self.node_type_names[BHMGNodeType.BASE.value]))

    @classmethod
    def build_from(cls, spec: BaseSpec, morphology: RobotMorphology) -> Self:
        """
        Factory constructor for BHMG feature extractor.

        Args:
            spec: Graph specification
            morphology: Robot morphology with node_metadata for joint classification.
                       Required to correctly partition joint data into joint, foot, and hand
                       nodes based on parser's classification, and to determine node counts.
        """
        return cls(spec, morphology)

    def _joint_indices_for_type(
        self, raw_data: dict[str, Any], node_type: str
    ) -> list[int]:
        joint_names = raw_data.get("joint_names")
        if not isinstance(joint_names, list):
            raise ValueError(
                "raw_data must include 'joint_names' list for feature mapping"
            )

        joint_name_items = cast(list[object], joint_names)
        joint_names_str = [str(name) for name in joint_name_items]
        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names_str)}
        indices: list[int] = []
        for name in self.node_type_names.get(node_type, []):
            if name in joint_name_to_idx:
                indices.append(joint_name_to_idx[name])
        return indices

    @staticmethod
    def _build_motor_scalar_features(
        raw_data: dict[str, Any],
        indices: list[int],
    ) -> torch.Tensor:
        if len(indices) == 0:
            return torch.zeros((0, 3), dtype=torch.float32)

        joint_pos = raw_data["joint_pos"].to(torch.float32)
        joint_vel = raw_data["joint_vel"].to(torch.float32)
        joint_torque = raw_data["joint_torque"].to(torch.float32)

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        pos = joint_pos.index_select(0, idx_tensor).unsqueeze(1)
        vel = joint_vel.index_select(0, idx_tensor).unsqueeze(1)
        torque = joint_torque.index_select(0, idx_tensor).unsqueeze(1)
        return torch.cat([pos, vel, torque], dim=1)

    def _extract_foot_kinematics(self, raw_data: dict[str, Any]) -> torch.Tensor:
        foot_pos = raw_data["foot_pos_w"].to(torch.float32)
        foot_vel = raw_data["foot_lin_vel_w"].to(torch.float32)

        if foot_pos.ndim != 2 or foot_vel.ndim != 2:
            raise ValueError(
                "foot_pos_w and foot_lin_vel_w must be rank-2 tensors "
                f"but got {tuple(foot_pos.shape)} and {tuple(foot_vel.shape)}"
            )
        if foot_pos.shape[1] != 3 or foot_vel.shape[1] != 3:
            raise ValueError(
                "foot_pos_w and foot_lin_vel_w must have last dim=3 "
                f"but got {tuple(foot_pos.shape)} and {tuple(foot_vel.shape)}"
            )

        expected = len(self.node_type_names[BHMGNodeType.FOOT.value])
        available = min(int(foot_pos.shape[0]), int(foot_vel.shape[0]))

        if expected <= 0:
            return torch.zeros((0, 6), dtype=torch.float32)

        out = torch.zeros((expected, 6), dtype=torch.float32)
        if available > 0:
            copy_count = min(expected, available)
            out[:copy_count, :3] = foot_pos[:copy_count, :]
            out[:copy_count, 3:] = foot_vel[:copy_count, :]
        return out

    def _extract_hand_kinematics(self, raw_data: dict[str, Any]) -> torch.Tensor:
        hand_pos_any = raw_data.get("hand_pos_w")
        hand_vel_any = raw_data.get("hand_lin_vel_w")
        expected = len(self.node_type_names[BHMGNodeType.HAND.value])

        if expected <= 0:
            return torch.zeros((0, 6), dtype=torch.float32)

        if not isinstance(hand_pos_any, torch.Tensor) or not isinstance(
            hand_vel_any, torch.Tensor
        ):
            raise ValueError(
                "BHMG hand nodes require hand kinematics inputs. "
                "Please provide raw_data['hand_pos_w'] and raw_data['hand_lin_vel_w'] "
                "with shape [num_hands, 3]."
            )

        hand_pos = hand_pos_any.to(torch.float32)
        hand_vel = hand_vel_any.to(torch.float32)

        if hand_pos.ndim != 2 or hand_vel.ndim != 2:
            raise ValueError(
                "hand_pos_w and hand_lin_vel_w must be rank-2 tensors "
                f"but got {tuple(hand_pos.shape)} and {tuple(hand_vel.shape)}"
            )
        if hand_pos.shape[1] != 3 or hand_vel.shape[1] != 3:
            raise ValueError(
                "hand_pos_w and hand_lin_vel_w must have last dim=3 "
                f"but got {tuple(hand_pos.shape)} and {tuple(hand_vel.shape)}"
            )

        available = min(int(hand_pos.shape[0]), int(hand_vel.shape[0]))
        out = torch.zeros((expected, 6), dtype=torch.float32)
        if available > 0:
            copy_count = min(expected, available)
            out[:copy_count, :3] = hand_pos[:copy_count, :]
            out[:copy_count, 3:] = hand_vel[:copy_count, :]
        return out

    def feature_type_layout(
        self,
    ) -> dict[str, list[tuple[int, int, BaseFeatureType]]]:
        return {
            BHMGNodeType.BASE.value: [
                (0, 3, BaseFeatureType.VECTOR_3D),
                (3, 6, BaseFeatureType.PSEUDOVECTOR_3D),
            ],
            BHMGNodeType.JOINT.value: [
                (0, 3, BaseFeatureType.SCALAR),
            ],
            BHMGNodeType.FOOT.value: [
                (0, 3, BaseFeatureType.VECTOR_3D),
                (3, 6, BaseFeatureType.VECTOR_3D),
            ],
            BHMGNodeType.HAND.value: [
                (0, 3, BaseFeatureType.VECTOR_3D),
                (3, 6, BaseFeatureType.VECTOR_3D),
            ],
        }

    def extract(self, raw_data: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Extract node features from raw timestep data for BHMG.

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

        # Extract hand features (hand kinematics)
        hand_features = self._extract_hand_features(raw_data)

        return {
            BHMGNodeType.BASE.value: base_features,
            BHMGNodeType.JOINT.value: joint_features,
            BHMGNodeType.FOOT.value: foot_features,
            BHMGNodeType.HAND.value: hand_features,
        }

    def _extract_base_features(self, raw_data: dict[str, Any]) -> torch.Tensor:
        """
        Extract base node features (IMU only).

        INPUT:
        - raw_data: Dict containing raw sensor data for one timestep

        OUTPUT:
        - torch.Tensor of shape [num_base_nodes, 6] (lin_acc + ang_vel)
        """
        imu_lin_acc = raw_data["imu_lin_acc"].to(torch.float32)
        imu_ang_vel = raw_data["imu_ang_vel"].to(torch.float32)
        base = torch.cat([imu_lin_acc, imu_ang_vel], dim=0).unsqueeze(0)
        return base.repeat(self.base_count, 1)

    def _extract_joint_features(self, raw_data: dict[str, Any]) -> torch.Tensor:
        """
        Extract joint node features (compact motor scalars).

        INPUT:
        - raw_data: Dict containing raw sensor data for one timestep

        OUTPUT:
        - torch.Tensor of shape [num_joint_nodes, 3] (pos, vel, torque)
        """
        indices = self._joint_indices_for_type(raw_data, BHMGNodeType.JOINT.value)
        return self._build_motor_scalar_features(raw_data, indices)

    def _extract_foot_features(self, raw_data: dict[str, Any]) -> torch.Tensor:
        """
        Extract foot node features (foot kinematics).

        INPUT:
        - raw_data: Dict containing raw sensor data for one timestep

        OUTPUT:
        - torch.Tensor of shape [num_foot_nodes, 6] (foot_pos_w, foot_lin_vel_w)
        """
        return self._extract_foot_kinematics(raw_data)

    def _extract_hand_features(self, raw_data: dict[str, Any]) -> torch.Tensor:
        """
        Extract hand node features (hand kinematics).

        INPUT:
        - raw_data: Dict containing raw sensor data for one timestep

        OUTPUT:
        - torch.Tensor of shape [num_hand_nodes, 6] (hand position xyz, hand linear velocity xyz)
        """
        return self._extract_hand_kinematics(raw_data)


BaseFeature.register(SpecType.BHMG)(BHMGFeatureExtractor)  # type: ignore[arg-type]
