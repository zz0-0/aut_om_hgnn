"""Streaming proprioceptive inference for trained OM-HGNN checkpoints.

Wraps a checkpoint into a stateful estimator that maintains its own history
buffer, so it can be called once per control step from a locomotion controller
or a hardware bridge without touching the offline dataset pipeline.
"""

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config.train_config import TrainConfig
from src.graph.spec.base_spec import BaseSpec
from src.graph.parser.base_parser import BaseParser
from src.graph.feature.base_feature import BaseFeature
from src.model.architecture.base_model import BaseModel
from src.model.training.base_lit_model import BaseLitModel
from src.config.train_enum import OutputType

SINGLE_STEP_VECTOR_KEYS = (
    "joint_pos",
    "joint_vel",
    "joint_torque",
    "imu_lin_acc",
    "imu_ang_vel",
)
FOOT_VECTOR_KEYS = ("foot_pos", "foot_lin_vel")
HAND_VECTOR_KEYS = ("hand_pos", "hand_lin_vel")


class StreamingEstimator:
    """Stateful single-sample estimator with an internal history buffer."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        joint_names: list[str],
        device: str | torch.device = "cpu",
        repo_root: str | Path | None = None,
    ):
        self.config = TrainConfig.build_from(str(config_path))
        if repo_root is not None:
            parser_path = Path(self.config.parser_path)
            if not parser_path.is_absolute():
                self.config.parser_path = Path(repo_root) / parser_path
        self.device = torch.device(device)
        if self.config.output_types:
            self.output_types: list[OutputType] = list(self.config.output_types)
        else:
            self.output_types = [self.config.output_type]
        self.output_type: OutputType = self.output_types[0]
        self.history_length = int(self.config.history_length)
        self.joint_names = [str(name) for name in joint_names]

        self.spec = BaseSpec.create_spec(
            self.config.spec_type, self.config.symmetry_type
        )
        parser = BaseParser.create_parser(
            self.config.robot_type,
            self.config.model_type,
            self.spec,
            self.config.parser_path,
        )
        self.morphology = parser.parse()
        self.feature_extractor = BaseFeature.create_extractor(
            spec=self.spec, morphology=self.morphology
        )

        model = BaseModel.create_model(train_config=self.config, spec=self.spec)
        lit_model = BaseLitModel.create_lit_model(
            model=model, spec=self.spec, train_config=self.config
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        lit_model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

        self.edge_index_dict = {
            key: value.to(self.device)
            for key, value in self.morphology.edge_index_dict.items()
        }
        self.buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=self.history_length)

    def reset(self) -> None:
        """Clear the history buffer (call on episode reset)."""
        self.buffer.clear()

    def _to_tensor(self, value: Any) -> torch.Tensor:
        array = np.asarray(value, dtype=np.float32)
        return torch.from_numpy(array.copy())

    def _single_step_features(self, frame: dict[str, Any]) -> dict[str, torch.Tensor]:
        raw_data: dict[str, Any] = {"joint_names": self.joint_names}
        for key in SINGLE_STEP_VECTOR_KEYS + FOOT_VECTOR_KEYS + HAND_VECTOR_KEYS:
            if key in frame:
                raw_data[key] = self._to_tensor(frame[key])
        return self.feature_extractor.extract(raw_data)

    def build_history_features(
        self, frame: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Update the buffer with a new frame and stack the history window."""
        features = self._single_step_features(frame)
        if not self.buffer:
            for _ in range(self.history_length):
                self.buffer.append(features)
        else:
            self.buffer.append(features)
        return {
            node_type: torch.cat(
                [step_features[node_type] for step_features in self.buffer], dim=1
            ).to(self.device)
            for node_type in features
        }

    @torch.no_grad()
    def step_all(self, frame: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run one inference step and return predictions for every configured head.

        INPUT:
        - frame: single-step proprioception with numpy arrays, keys:
          joint_pos, joint_vel, joint_torque, imu_lin_acc, imu_ang_vel
          (body/base frame) and foot_pos, foot_lin_vel (optional hand_pos,
          hand_lin_vel) in base frame.

        OUTPUT:
        - Dict mapping output type name to prediction array.
        """
        x_dict = self.build_history_features(frame)
        outputs = self.model(x_dict, self.edge_index_dict)
        predictions: dict[str, np.ndarray] = {}
        for output_type in self.output_types:
            prediction = outputs[output_type]
            if output_type == OutputType.CONTACT:
                prediction = torch.sigmoid(prediction)
            predictions[output_type.value] = prediction.detach().cpu().numpy()
        return predictions

    @torch.no_grad()
    def step(self, frame: dict[str, Any]) -> np.ndarray:
        """Run one inference step and return the primary output prediction."""
        return self.step_all(frame)[self.output_type.value]
