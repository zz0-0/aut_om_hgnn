"""Unitree G1 dataset with history-agnostic processed cache.

This dataset builds heterogeneous graph samples from raw trajectory memmaps and stores
preprocessed single-step tensors to one shared cache folder:

    processed/<dataset_name>/

Why this matters:
- Different ``history_length`` values should reuse the same underlying cache.
- Expanding and storing every overlapping history window causes huge disk usage.
- We therefore cache only single-step features and build history windows on-the-fly
    in ``get()`` for the requested ``history_length``.

The processing pipeline is intentionally verbose so long preprocessing runs do not look
stuck and users can track progress and ETA in the terminal.
"""

import json
import os.path as osp
from pathlib import Path
from typing import Any, Self, cast
import logging
import time
import numpy as np
import torch

from torch_geometric.data import HeteroData  # type: ignore

from src.graph.feature.base_feature import BaseFeature
from src.graph.spec.base_spec import BaseSpec
from src.config.train_config import RobotType
from src.data.dataset.base_dataset import BaseDataset
from src.graph.morphology.base_morphology import RobotMorphology


class UnitreeG1Dataset(BaseDataset):
    """History-agnostic dataset for Unitree robots backed by NumPy memmaps."""

    def __init__(
        self,
        dataset_path: Path,
        morphology: RobotMorphology,
        spec: BaseSpec,
        history_length: int = 1,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.stem
        self.morphology = morphology
        self.spec = spec
        self.history_length = int(history_length)
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        self.root = str(self.dataset_path.parent)
        self.logger = logging.getLogger(__name__)
        self._ensure_logger_configured()
        self._processed_arrays: dict[str, np.memmap] = {}
        # Global sample index -> (env_idx, episode_idx, time_idx)
        self.sample_lut: np.ndarray = np.empty((0, 3), dtype=np.int32)
        self.base_sample_lut: np.ndarray = np.empty((0, 3), dtype=np.int32)
        self.sample_end_indices: np.ndarray = np.empty((0,), dtype=np.int64)
        self.num_samples = 0
        self._logged_feature_mode = False
        self.node_feature_dims = self.spec.node_types_with_history(self.history_length)
        self.base_node_feature_dims = self.spec.node_types_with_history(1)
        self.node_counts = {
            node_type: len(self.morphology.node_type_usd_node_dict.get(node_type, []))
            for node_type in self.node_feature_dims
        }
        if "base" in self.node_counts:
            self.node_counts["base"] = max(1, self.node_counts["base"])

        self.feature_extractor = BaseFeature.create_extractor(
            spec=spec, morphology=morphology
        )
        self.feature_type_layout = (
            self.feature_extractor.feature_type_layout_serialized()
        )

        self.logger.info(
            "Dataset init: dataset=%s | history_length=%d | processed_dir=%s",
            self.dataset_name,
            self.history_length,
            self.processed_dir,
        )
        self.logger.info("Dataset node feature dims: %s", self.node_feature_dims)
        self.logger.info(
            "Dataset base node feature dims: %s", self.base_node_feature_dims
        )
        self.logger.info("Dataset node counts: %s", self.node_counts)

        self._load_or_process_dataset()

        super().__init__(root=self.root)

    @classmethod
    def build_from(
        cls,
        dataset_path: Path,
        morphology: RobotMorphology,
        spec: BaseSpec,
        history_length: int = 1,
    ) -> Self:
        """
        Factory constructor for G1Dataset.

        INPUT:
        - dataset_path: Path to Numpy memmap file
        - morphology: Robot morphology from parser
        - spec: BHMG specification

        OUTPUT:
        - Initialized G1Dataset ready to use as PyTorch Dataset
        """
        return cls(
            dataset_path=dataset_path,
            morphology=morphology,
            spec=spec,
            history_length=history_length,
        )

    def _ensure_logger_configured(self) -> None:
        """Ensure INFO logs are visible even when caller did not configure logging.

        We only attach a local stream handler when neither this logger nor the root
        logger has handlers configured, to avoid duplicate logs in normal training runs.
        """
        self.logger.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        has_any_handler = bool(self.logger.handlers) or bool(root_logger.handlers)
        if not has_any_handler:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _load_or_process_dataset(self):
        """Load compatible processed cache, or rebuild it from raw memmaps."""
        self.processed_data_path = Path(self.processed_paths[0])
        self.logger.info(
            "Resolved processed manifest path: %s", self.processed_data_path
        )
        if self.processed_data_path.exists():
            manifest = json.loads(self.processed_data_path.read_text())
            if self._is_manifest_compatible(manifest):
                self.logger.info(
                    "Loading processed memmap cache from %s", self.processed_dir
                )
                self._open_processed_cache(manifest)
            else:
                self.logger.info(
                    "Processed cache at %s is incompatible with current config. "
                    "Reprocessing dataset.",
                    self.processed_dir,
                )
                self.process()
        else:
            self.logger.info(
                "No processed cache found at %s. Processing raw data from %s",
                self.processed_data_path,
                self.dataset_path,
            )
            self.process()

        self._configure_runtime_sample_lut()

    def _is_manifest_compatible(self, manifest: dict[str, Any]) -> bool:
        """Validate whether a processed cache manifest matches current dataset settings."""
        manifest_dims = {
            key: int(value)
            for key, value in manifest.get("node_feature_dims", {}).items()
        }
        manifest_counts = {
            key: int(value) for key, value in manifest.get("node_counts", {}).items()
        }

        if manifest_dims != self.base_node_feature_dims:
            return False

        if manifest_counts != self.node_counts:
            return False

        return True

    @property
    def processed_dir(self):
        """Use one shared processed cache directory per dataset path."""
        return osp.join(self.root, "processed", self.dataset_name)

    @property
    def raw_file_names(self):
        """PyG Dataset contract: raw files expected to exist for this dataset."""
        return ["metadata.json"]

    @property
    def processed_file_names(self):
        """PyG Dataset contract: sentinel processed file used as cache-ready flag."""
        return ["manifest.json"]

    def len(self) -> int:
        """Return total number of samples in dataset."""
        return self.num_samples

    def get(self, idx: int) -> HeteroData:
        """
        Get a single sample.

        INPUT:
        - idx: Sample index

        OUTPUT:
        - HeteroData sample with x_dict, edge_index_dict, y, and metadata
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {self.num_samples}"
            )

        data = HeteroData()
        history = self.history_length
        end_cache_idx = int(self.sample_end_indices[idx])
        start_cache_idx = end_cache_idx - history + 1

        for node_type in self.node_feature_dims:
            array_key = f"x_{node_type}"
            if array_key in self._processed_arrays:
                node_array = self._processed_arrays[array_key]
                if history == 1:
                    node_x_np = np.asarray(node_array[end_cache_idx], dtype=np.float32)
                else:
                    window_np = np.asarray(
                        node_array[start_cache_idx : end_cache_idx + 1],
                        dtype=np.float32,
                    )
                    node_x_np = np.transpose(window_np, (1, 0, 2)).reshape(
                        window_np.shape[1], -1
                    )
                data[node_type].x = self._to_tensor(node_x_np)

        available_node_types = set(self.node_feature_dims)
        for edge_key, edge_index in self.morphology.edge_index_dict.items():
            src_type, _, dst_type = edge_key
            if src_type in available_node_types and dst_type in available_node_types:
                data[edge_key].edge_index = edge_index

        data.y_contact = self._to_tensor(
            self._processed_arrays["y_contact"][end_cache_idx]
        ).unsqueeze(0)
        data.y_ground_reaction_force = self._to_tensor(
            self._processed_arrays["y_ground_reaction_force"][end_cache_idx]
        ).unsqueeze(0)
        data.y_center_of_mass = self._to_tensor(
            self._processed_arrays["y_center_of_mass"][end_cache_idx]
        ).unsqueeze(0)
        data.feature_type_layout = self.feature_type_layout

        env_idx, episode_idx, time_idx = self.sample_lut[idx].tolist()
        data.env_idx = int(env_idx)
        data.episode_idx = int(episode_idx)
        data.time_idx = int(time_idx)

        return data

    def _configure_runtime_sample_lut(self) -> None:
        """Build runtime sample indices for the requested history length.

        Cache stores single-step entries only. For ``history_length > 1``, we build
        valid end indices from contiguous runs in ``base_sample_lut``.
        """
        if self.base_sample_lut.size == 0:
            self.sample_end_indices = np.empty((0,), dtype=np.int64)
            self.sample_lut = np.empty((0, 3), dtype=np.int32)
            self.num_samples = 0
            return

        history = self.history_length
        if history == 1:
            self.sample_end_indices = np.arange(
                self.base_sample_lut.shape[0], dtype=np.int64
            )
        else:
            end_indices: list[int] = []
            base_lut = self.base_sample_lut
            total_rows = base_lut.shape[0]
            run_start = 0

            while run_start < total_rows:
                env_idx, episode_idx, time_idx = base_lut[run_start].tolist()
                run_end = run_start + 1
                prev_time = int(time_idx)

                while run_end < total_rows:
                    next_env, next_episode, next_time = base_lut[run_end].tolist()
                    if (
                        int(next_env) != int(env_idx)
                        or int(next_episode) != int(episode_idx)
                        or int(next_time) != prev_time + 1
                    ):
                        break
                    prev_time = int(next_time)
                    run_end += 1

                run_len = run_end - run_start
                if run_len >= history:
                    end_indices.extend(range(run_start + history - 1, run_end))

                run_start = run_end

            self.sample_end_indices = np.asarray(end_indices, dtype=np.int64)

        self.sample_lut = self.base_sample_lut[self.sample_end_indices].astype(np.int32)
        self.num_samples = int(self.sample_end_indices.shape[0])
        self.logger.info(
            "Runtime LUT ready: history_length=%d | samples=%d",
            self.history_length,
            self.num_samples,
        )

    def process(self):
        """Run full preprocessing from raw memmaps to processed cache files."""
        process_start = time.perf_counter()
        self.logger.info("Starting dataset preprocessing pipeline")
        self.logger.info("[1/5] Loading metadata and raw memmap fields...")
        self.metadata = self._load_metadata()
        raw_fields = self._load_raw_fields(self.metadata)
        self._joint_names = [
            str(name) for name in self.metadata["index_maps"].get("joint_names", [])
        ]
        self.logger.info("Loaded %d joint names", len(self._joint_names))
        self.logger.info("[2/5] Building sample lookup table...")
        sample_lut = self._build_sample_lut(self.metadata)
        self.logger.info("[3/5] Materializing processed cache arrays...")
        arrays = self._materialize_processed_cache(raw_fields, sample_lut)
        self.logger.info("[4/5] Flushing processed arrays to disk...")
        self._flush_arrays(arrays)
        self.logger.info("[5/5] Writing manifest and opening cache...")
        manifest = self._build_manifest(arrays, num_samples=int(sample_lut.shape[0]))
        self.processed_data_path.write_text(json.dumps(manifest, indent=2))
        self._open_processed_cache()
        self.logger.info(
            "Dataset preprocessing completed in %s",
            self._format_duration(time.perf_counter() - process_start),
        )

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata.json that describes field files, shapes, and index maps."""
        metadata_path = self.dataset_path / "metadata.json"
        self.logger.info("Loading metadata from %s", metadata_path)
        return json.loads(metadata_path.read_text())

    def _load_raw_fields(self, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
        """Open all raw field memmaps declared in metadata without loading into RAM."""
        fields = {
            field_name: np.load(self.dataset_path / rel_path, mmap_mode="r")
            for field_name, rel_path in metadata["files"]["fields"].items()
        }
        self.logger.info(
            "Loaded %d raw fields: %s",
            len(fields),
            sorted(fields.keys()),
        )
        return fields

    def _build_sample_lut(self, metadata: dict[str, Any]) -> np.ndarray:
        """
        Build global sample lookup table.

        Each row stores (env_idx, episode_idx, time_idx) for one valid training sample.
        This lets __getitem__(idx) jump directly to raw/processed arrays without scanning.
        """
        valid_steps = np.load(
            self.dataset_path / metadata["files"]["valid_steps"], mmap_mode="r"
        )
        valid_steps_bool = np.asarray(valid_steps, dtype=bool)
        self.logger.info(
            "valid_steps shape=%s | cache_mode=single-step",
            tuple(valid_steps_bool.shape),
        )
        sample_lut = np.argwhere(valid_steps_bool).astype(np.int32)

        self.logger.info(
            "Processing %d valid single-step entries into processed memmap cache",
            int(sample_lut.shape[0]),
        )
        return sample_lut

    def _materialize_processed_cache(
        self,
        raw_fields: dict[str, np.ndarray],
        sample_lut: np.ndarray,
    ) -> dict[str, np.memmap]:
        """Create processed memmap files and fill them sample-by-sample."""
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        num_samples = int(sample_lut.shape[0])
        self.logger.info(
            "Preparing processed cache directory %s for %d samples",
            self.processed_dir,
            num_samples,
        )
        arrays = self._create_processed_arrays(num_samples)
        arrays["sample_lut"][:] = sample_lut

        if num_samples == 0:
            self.logger.warning(
                "No valid samples found. Writing empty processed cache."
            )
            return arrays

        report_every = min(50_000, max(1, num_samples // 20))
        self.logger.info("Cache write report interval: every %d samples", report_every)
        loop_start = time.perf_counter()

        for sample_idx, (env_idx, episode_idx, time_idx) in enumerate(sample_lut):
            raw_data = self._extract_raw_data(
                raw_fields,
                int(env_idx),
                int(episode_idx),
                int(time_idx),
                history_length=1,
            )
            x_dict = self._extract_features(raw_data, history_length=1)
            self._write_processed_sample(arrays, sample_idx, x_dict, raw_data)

            done = sample_idx + 1
            if done % report_every == 0 or done == num_samples:
                elapsed = max(time.perf_counter() - loop_start, 1e-6)
                rate = done / elapsed
                remaining = max(num_samples - done, 0)
                eta_seconds = remaining / max(rate, 1e-6)
                self.logger.info(
                    "Cache write progress: %d/%d (%.1f%%) | %.0f samples/s | ETA %s",
                    done,
                    num_samples,
                    100.0 * done / num_samples,
                    rate,
                    self._format_duration(eta_seconds),
                )

        return arrays

    def _write_processed_sample(
        self,
        arrays: dict[str, np.memmap],
        sample_idx: int,
        x_dict: dict[str, torch.Tensor],
        raw_data: dict[str, torch.Tensor],
    ) -> None:
        for node_type in self.base_node_feature_dims:
            array_key = f"x_{node_type}"
            if array_key in arrays:
                arrays[array_key][sample_idx] = x_dict[node_type].cpu().numpy()

        contact_states = raw_data["contact_states"]
        if contact_states.ndim > 1:
            contact_states = contact_states[-1]
        arrays["y_contact"][sample_idx] = contact_states.cpu().numpy()

        contact_forces = raw_data["contact_forces"]
        if contact_forces.ndim > 2:
            contact_forces = contact_forces[-1]
        arrays["y_ground_reaction_force"][sample_idx] = (
            contact_forces.reshape(-1).cpu().numpy()
        )

        base_lin_vel = raw_data["base_lin_vel"]
        base_ang_vel = raw_data["base_ang_vel"]
        if base_lin_vel.ndim > 1:
            base_lin_vel = base_lin_vel[-1]
        if base_ang_vel.ndim > 1:
            base_ang_vel = base_ang_vel[-1]
        arrays["y_center_of_mass"][sample_idx] = (
            torch.cat([base_lin_vel, base_ang_vel]).cpu().numpy()
        )

    def _flush_arrays(self, arrays: dict[str, np.memmap]) -> None:
        """Flush all processed memmap buffers to disk."""
        self.logger.info("Flushing %d processed arrays to disk", len(arrays))
        for arr in arrays.values():
            arr.flush()

    def _build_manifest(
        self,
        arrays: dict[str, np.memmap],
        num_samples: int,
    ) -> dict[str, Any]:
        """Build manifest metadata required to reopen processed cache quickly."""
        return {
            "version": 1,
            "num_samples": num_samples,
            "history_length": 1,
            "node_feature_dims": self.base_node_feature_dims,
            "node_counts": self.node_counts,
            "arrays": {
                name: {
                    "filename": f"{name}.npy",
                    "shape": list(memmap.shape),
                    "dtype": str(memmap.dtype),
                }
                for name, memmap in arrays.items()
            },
        }

    def _create_processed_arrays(self, num_samples: int) -> dict[str, np.memmap]:
        """Allocate output memmaps with exact shapes based on spec and morphology."""
        arrays: dict[str, np.memmap] = {}

        for node_type, feature_dim in self.base_node_feature_dims.items():
            num_nodes = self.node_counts.get(node_type, 0)
            arrays[f"x_{node_type}"] = np.memmap(
                Path(self.processed_dir) / f"x_{node_type}.npy",
                dtype=np.float32,
                mode="w+",
                shape=(num_samples, num_nodes, feature_dim),
            )
            self.logger.info(
                "Allocated x_%s with shape=%s",
                node_type,
                (num_samples, num_nodes, feature_dim),
            )

        contact_dim = int(self.metadata["field_shapes"]["contact_states"][-1])
        foot_dim = int(self.metadata["field_shapes"]["contact_forces"][-2])
        grf_dim = foot_dim * 3

        arrays["y_contact"] = np.memmap(
            Path(self.processed_dir) / "y_contact.npy",
            dtype=np.float32,
            mode="w+",
            shape=(num_samples, contact_dim),
        )
        arrays["y_ground_reaction_force"] = np.memmap(
            Path(self.processed_dir) / "y_ground_reaction_force.npy",
            dtype=np.float32,
            mode="w+",
            shape=(num_samples, grf_dim),
        )
        arrays["y_center_of_mass"] = np.memmap(
            Path(self.processed_dir) / "y_center_of_mass.npy",
            dtype=np.float32,
            mode="w+",
            shape=(num_samples, 6),
        )
        arrays["sample_lut"] = np.memmap(
            Path(self.processed_dir) / "sample_lut.npy",
            dtype=np.int32,
            mode="w+",
            shape=(num_samples, 3),
        )

        self.logger.info(
            "Allocated target arrays: y_contact=%s | y_grf=%s | y_com=%s | sample_lut=%s",
            (num_samples, contact_dim),
            (num_samples, grf_dim),
            (num_samples, 6),
            (num_samples, 3),
        )

        return arrays

    def _open_processed_cache(self, manifest: dict[str, Any] | None = None) -> None:
        """Open processed memmaps declared in manifest and validate compatibility."""
        manifest_data: dict[str, Any]
        if manifest is None:
            manifest_data = json.loads(self.processed_data_path.read_text())
        else:
            manifest_data = manifest

        base_num_samples = int(manifest_data["num_samples"])
        manifest_dims = {
            str(key): int(value)
            for key, value in cast(
                dict[str, Any], manifest_data["node_feature_dims"]
            ).items()
        }
        manifest_counts = {
            str(key): int(value)
            for key, value in cast(dict[str, Any], manifest_data["node_counts"]).items()
        }

        if manifest_dims != self.base_node_feature_dims:
            raise ValueError(
                "Processed cache node_feature_dims are incompatible with current config. "
                f"Expected {self.base_node_feature_dims}, got {manifest_dims}."
            )
        if manifest_counts != self.node_counts:
            raise ValueError(
                "Processed cache node_counts are incompatible with current morphology. "
                f"Expected {self.node_counts}, got {manifest_counts}."
            )

        self._processed_arrays = {}
        for name, array_info in cast(dict[str, Any], manifest_data["arrays"]).items():
            info = cast(dict[str, Any], array_info)
            self._processed_arrays[name] = np.memmap(
                Path(self.processed_dir) / str(info["filename"]),
                dtype=np.dtype(str(info["dtype"])),
                mode="r",
                shape=tuple(cast(list[int], info["shape"])),
            )
            self.logger.info(
                "Opened cache array %s with shape=%s dtype=%s",
                name,
                tuple(cast(list[int], info["shape"])),
                str(info["dtype"]),
            )

        self.base_sample_lut = np.asarray(
            self._processed_arrays["sample_lut"], dtype=np.int32
        )
        self.logger.info("Base cache ready: num_samples=%d", base_num_samples)

    def _extract_raw_data(
        self,
        raw_fields: dict[str, np.ndarray],
        env_idx: int,
        episode_idx: int,
        time_idx: int,
        history_length: int = 1,
    ) -> dict[str, Any]:
        start_idx = time_idx - history_length + 1
        end_idx = time_idx + 1

        # Support both new field names (aligned with v3 node types) and old names
        # for backward compatibility with previously exported datasets.
        _torque_key = (
            "joint_torque" if "joint_torque" in raw_fields else "joint_torques"
        )
        _imu_acc_key = "imu_lin_acc" if "imu_lin_acc" in raw_fields else "imu_lin_acc_b"
        _imu_vel_key = "imu_ang_vel" if "imu_ang_vel" in raw_fields else "imu_ang_vel_b"

        extracted: dict[str, Any] = {
            "joint_pos": self._to_tensor(
                raw_fields["joint_pos"][env_idx, episode_idx, start_idx:end_idx]
            ),
            "joint_vel": self._to_tensor(
                raw_fields["joint_vel"][env_idx, episode_idx, start_idx:end_idx]
            ),
            "joint_torque": self._to_tensor(
                raw_fields[_torque_key][env_idx, episode_idx, start_idx:end_idx]
            ),
            "imu_lin_acc": self._to_tensor(
                raw_fields[_imu_acc_key][env_idx, episode_idx, start_idx:end_idx]
            ),
            "imu_ang_vel": self._to_tensor(
                raw_fields[_imu_vel_key][env_idx, episode_idx, start_idx:end_idx]
            ),
            "contact_states": self._to_tensor(
                raw_fields["contact_states"][env_idx, episode_idx, start_idx:end_idx]
            ),
            "contact_forces": self._to_tensor(
                raw_fields["contact_forces"][env_idx, episode_idx, start_idx:end_idx]
            ),
            "foot_pos_w": self._to_tensor(
                raw_fields["foot_pos_w"][env_idx, episode_idx, start_idx:end_idx]
            ),
            "foot_lin_vel_w": self._to_tensor(
                raw_fields["foot_lin_vel_w"][env_idx, episode_idx, start_idx:end_idx]
            ),
            "base_lin_vel": self._to_tensor(
                raw_fields["root_com_lin_vel_w"][
                    env_idx, episode_idx, start_idx:end_idx
                ]
            ),
            "base_ang_vel": self._to_tensor(
                raw_fields["root_com_ang_vel_w"][
                    env_idx, episode_idx, start_idx:end_idx
                ]
            ),
            "joint_names": self._joint_names,
        }

        if "hand_pos_w" in raw_fields:
            extracted["hand_pos_w"] = self._to_tensor(
                raw_fields["hand_pos_w"][env_idx, episode_idx, start_idx:end_idx]
            )
        if "hand_lin_vel_w" in raw_fields:
            extracted["hand_lin_vel_w"] = self._to_tensor(
                raw_fields["hand_lin_vel_w"][
                    env_idx,
                    episode_idx,
                    start_idx:end_idx,
                ]
            )

        return extracted

    def _extract_features(
        self,
        raw_data: dict[str, Any],
        history_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract per-node features and stack history dimension when requested."""
        effective_history = (
            self.history_length if history_length is None else history_length
        )

        if not self._logged_feature_mode:
            mode = "single-step" if effective_history == 1 else "history-stacked"
            self.logger.info(
                "Feature extraction mode: %s (history_length=%d)",
                mode,
                effective_history,
            )
            self._logged_feature_mode = True

        if effective_history == 1:
            single_step_raw_data = {
                key: (
                    value[0]
                    if isinstance(value, torch.Tensor) and value.ndim > 1
                    else value
                )
                for key, value in raw_data.items()
            }
            x_dict = self.feature_extractor.extract(single_step_raw_data)
        else:
            stacked_by_node_type: dict[str, list[torch.Tensor]] = {
                node_type: [] for node_type in self.base_node_feature_dims
            }

            for t in range(effective_history):
                step_raw_data = {
                    key: (
                        value[t]
                        if isinstance(value, torch.Tensor) and value.ndim > 1
                        else value
                    )
                    for key, value in raw_data.items()
                }
                step_x_dict = self.feature_extractor.extract(step_raw_data)
                for node_type, node_x in step_x_dict.items():
                    stacked_by_node_type[node_type].append(node_x)

            x_dict = {
                node_type: torch.cat(time_slices, dim=1)
                for node_type, time_slices in stacked_by_node_type.items()
            }

        expected_feature_dims = (
            self.base_node_feature_dims
            if effective_history == 1
            else self.spec.node_types_with_history(effective_history)
        )

        validated: dict[str, torch.Tensor] = {}
        for node_type, feature_dim in expected_feature_dims.items():
            tensor = x_dict.get(node_type)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"Feature extractor output for {node_type} is not a Tensor"
                )

            expected_nodes = self.node_counts.get(node_type, 0)
            if tensor.shape != (expected_nodes, feature_dim):
                raise ValueError(
                    f"Feature shape mismatch for {node_type}. "
                    f"Expected {(expected_nodes, feature_dim)}, got {tuple(tensor.shape)}"
                )

            validated[node_type] = tensor.to(torch.float32)

        return validated

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        return torch.tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32)

    def _zero_feature_dict(self) -> dict[str, torch.Tensor]:
        return {
            node_type: torch.zeros(
                (self.node_counts.get(node_type, 0), feature_dim),
                dtype=torch.float32,
            )
            for node_type, feature_dim in self.node_feature_dims.items()
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = int(max(0.0, seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes > 0:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"


class UnitreeG1Dataset23DOF(UnitreeG1Dataset):
    """Dataset for Unitree G1 23-DOF variant."""

    pass


BaseDataset.register(RobotType.UNITREE_G1_29DOF)(UnitreeG1Dataset)  # type: ignore[arg-type]
BaseDataset.register(RobotType.UNITREE_G1_23DOF)(UnitreeG1Dataset23DOF)  # type: ignore[arg-type]
