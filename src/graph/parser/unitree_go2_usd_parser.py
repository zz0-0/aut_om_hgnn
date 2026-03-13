from pathlib import Path
from typing import Self

from pxr import Usd, UsdPhysics  # type: ignore
import torch

from src.graph.spec.qhmg import QHMGNodeType
from src.graph.morphology.base_morphology import RobotMorphology
from src.graph.spec.base_spec import BaseSpec
from src.config.train_enum import ModelType, RobotType
from src.config.batch_schema import edge_index_dict_type
from src.graph.parser.base_parser import BaseParser


class UnitreeGO2USDParser(BaseParser):
    """
    Parse Unitree GO2 robot morphology from USD (Pixar Universal Scene Description) files.

    WORKFLOW:
    1. Load base USD (contains robot hierarchy/geometry)
    2. Load physics USD (contains joint definitions)
    3. Extract joint parents/children from physics stage
    4. Build node list from joints (base, joints, feet)
    5. Build edge connectivity based on joint hierarchy
    6. Create edge_indices tensors for each edge type
    7. Return RobotMorphology

    FILE STRUCTURE:
    - Base USD: go2.usd (robot geometry/structure)
    - Physics USD: in configuration/ subfolder with _physics.usd suffix

    NODE IDENTIFICATION:
    - Base: root link (torso/pelvis)
    - Joints: all physics.Joint prims in hierarchy
    - Feet: child links of ankle joints (contains "foot" or "ankle" in name)
    """

    def __init__(self, model_type: ModelType, spec: BaseSpec, parser_path: Path):
        """
        Initialize USD parser.

        INPUT:
        - model_type: Model type (e.g., MS_HGNN)
        - spec: QHMG specification
        - parser_path: Path to base USD file
        """
        super().__init__()
        self.model_type = model_type
        self.spec = spec
        self.parser_path = Path(parser_path)
        self.physics_path = self.find_physics_usd_path()

    @classmethod
    def build_from(
        cls, model_type: ModelType, spec: BaseSpec, parser_path: Path
    ) -> Self:
        """
        Factory constructor for USDParser.

        INPUT:
        - spec: QHMG specification
        - parser_path: Path to base USD file

        OUTPUT:
        - Initialized USDParser ready to call parse()

        EXAMPLE:
        parser = USDParser.build_from(spec, "usd/go2.usd")
        morphology = parser.parse()
        """
        return cls(model_type, spec, parser_path)

    def find_physics_usd_path(self) -> Path:
        """
        Find physics USD file corresponding to base USD.

        LOGIC:
        - Base: go2.usd
        - Physics: configuration/go2_description_physics.usd

        OUTPUT:
        - Path to physics USD file

        RAISES:
        - FileNotFoundError: If physics USD not found
        """
        base_stem = self.parser_path.stem

        # Try configuration/ subfolder first
        physics_path = (
            self.parser_path.parent
            / "configuration"
            / f"{base_stem}_description_physics.usd"
        )
        if physics_path.exists():
            return physics_path

        # Try same directory
        physics_path = self.parser_path.parent / f"{base_stem}_description_physics.usd"
        if physics_path.exists():
            return physics_path

        raise FileNotFoundError(
            f"Could not find physics USD for {self.parser_path}\n"
            f"Searched for:\n"
            f"  {self.parser_path.parent / 'configuration' / f'{base_stem}_description_physics.usd'}\n"
            f"  {self.parser_path.parent / f'{base_stem}_description_physics.usd'}"
        )

    def parse(self) -> RobotMorphology:
        """
        Parse USD file and return robot morphology.

        OUTPUT:
        - RobotMorphology with edge_indices and edge_attrs tensors

        RAISES:
        - FileNotFoundError: If USD files don't exist
        - ValueError: If USD parsing fails
        """
        base_stage = Usd.Stage.Open(str(self.parser_path))  # type: ignore
        physics_stage = Usd.Stage.Open(str(self.physics_path))  # type: ignore

        if not base_stage or not physics_stage:
            raise ValueError(
                f"Failed to load USD files:\n"
                f"  Base: {self.parser_path}\n"
                f"  Physics: {self.physics_path}"
            )

        node_type_usd_node_dict = self._extract_usd_to_node_types(physics_stage)  # type: ignore
        self._node_type_usd_node_dict = node_type_usd_node_dict
        node_type_usd_node_index_dict = self._build_index_dict(node_type_usd_node_dict)
        edge_index_dict = self._build_edge_index_dict(node_type_usd_node_index_dict)

        morphology = RobotMorphology(
            node_type_usd_node_dict, node_type_usd_node_index_dict, edge_index_dict
        )
        return morphology

    def _extract_usd_to_node_types(
        self, physics_stage: Usd.Stage  # type: ignore
    ) -> dict[str, list[str]]:
        node_type: dict[str, list[str]] = {
            QHMGNodeType.BASE.value: [],
            QHMGNodeType.JOINT.value: [],
            QHMGNodeType.FOOT.value: [],
        }

        # Extract joints from physics USD
        self.joint_parents: dict[str, str] = {}  # joint_name -> parent_link_name
        self.joint_children: dict[str, str] = {}  # joint_name -> child_link_name
        self.joint_names: list[str] = []

        for prim in physics_stage.Traverse():  # type: ignore
            if prim.IsA(UsdPhysics.Joint):  # type: ignore
                joint_name: str = prim.GetPath().name  # type: ignore

                # Extract parent/child links from physics joint
                body0_rel = prim.GetRelationship("physics:body0")  # type: ignore
                body1_rel = prim.GetRelationship("physics:body1")  # type: ignore

                if not body0_rel.GetTargets() or not body1_rel.GetTargets():  # type: ignore
                    continue

                parent_link: str = body0_rel.GetTargets()[0].name  # type: ignore
                child_link: str = body1_rel.GetTargets()[0].name  # type: ignore

                self.joint_parents[joint_name] = parent_link
                self.joint_children[joint_name] = child_link
                self.joint_names.append(joint_name)  # type: ignore

        # Build joint connectivity graph using parent/child link matching.
        # child_joint.parent_link == parent_joint.child_link => parent_joint -> child_joint
        joint_parent_map: dict[str, str | None] = {
            joint_name: None for joint_name in self.joint_names
        }
        joint_children_map: dict[str, list[str]] = {
            joint_name: [] for joint_name in self.joint_names
        }

        for child_joint, parent_link in self.joint_parents.items():
            for parent_joint, child_link in self.joint_children.items():
                if parent_link == child_link:
                    joint_parent_map[child_joint] = parent_joint
                    joint_children_map[parent_joint].append(child_joint)
                    break

        # Base is the unique parent link of root joints.
        root_joints = [
            joint_name
            for joint_name in self.joint_names
            if joint_parent_map[joint_name] is None
        ]
        root_parent_links = {
            self.joint_parents[joint_name] for joint_name in root_joints
        }
        if len(root_parent_links) == 1:
            node_type[QHMGNodeType.BASE.value] = [next(iter(root_parent_links))]
        elif len(root_parent_links) == 0:
            raise ValueError("No root parent link found for GO2 base node.")
        else:
            raise ValueError(
                f"Expected single root link for GO2 base, got {len(root_parent_links)}: {sorted(root_parent_links)}"
            )

        # Classify kinematic joints: internal nodes are JOINT, leaf ankle/foot nodes are FOOT.
        self._joint_parent_map = joint_parent_map
        self._joint_children_map = joint_children_map
        self._root_joints = root_joints

        leg_prefixes = ("FL_", "FR_", "RL_", "RR_")

        for joint_name in self.joint_names:
            if not joint_name.startswith(leg_prefixes):
                continue
            children = joint_children_map[joint_name]
            name_lower = joint_name.lower()

            if len(children) > 0:
                node_type[QHMGNodeType.JOINT.value].append(joint_name)
            elif "foot" in name_lower or "ankle" in name_lower:
                node_type[QHMGNodeType.FOOT.value].append(joint_name)

        self._validate_go2_node_types(node_type)

        return node_type

    def _validate_go2_node_types(self, node_type: dict[str, list[str]]) -> None:
        base_nodes = node_type.get(QHMGNodeType.BASE.value, [])
        joint_nodes = node_type.get(QHMGNodeType.JOINT.value, [])
        foot_nodes = node_type.get(QHMGNodeType.FOOT.value, [])

        if len(base_nodes) != 1:
            raise ValueError(
                f"GO2 parser expected exactly 1 base node, got {len(base_nodes)}: {base_nodes}"
            )

        if len(joint_nodes) != 12:
            raise ValueError(
                "GO2 parser expected exactly 12 locomotion joints "
                f"(3 per leg), got {len(joint_nodes)}: {joint_nodes}"
            )

        if len(foot_nodes) != 4:
            raise ValueError(
                f"GO2 parser expected exactly 4 feet, got {len(foot_nodes)}: {foot_nodes}"
            )

        if len(set(joint_nodes)) != len(joint_nodes):
            raise ValueError("GO2 parser produced duplicate joint node names.")

        if len(set(foot_nodes)) != len(foot_nodes):
            raise ValueError("GO2 parser produced duplicate foot node names.")

    def _build_index_dict(
        self, node_type_usd_node_dict: dict[str, list[str]]
    ) -> dict[str, list[int]]:
        node_type_index_dict: dict[str, list[int]] = {}
        for node_type, usd_nodes in node_type_usd_node_dict.items():
            node_type_index_dict[node_type] = list(range(len(usd_nodes)))
        return node_type_index_dict

    def _build_edge_index_dict(
        self, node_type_index_dict: dict[str, list[int]]
    ) -> edge_index_dict_type:
        edge_index_dict: edge_index_dict_type = {}

        node_names = self._node_type_usd_node_dict
        joint_names = node_names.get(QHMGNodeType.JOINT.value, [])
        foot_names = node_names.get(QHMGNodeType.FOOT.value, [])

        joint_index_by_name = {name: idx for idx, name in enumerate(joint_names)}
        foot_index_by_name = {name: idx for idx, name in enumerate(foot_names)}

        root_joint_indices = [
            joint_index_by_name[name]
            for name in getattr(self, "_root_joints", [])
            if name in joint_index_by_name
        ]

        joint_to_joint_src: list[int] = []
        joint_to_joint_dst: list[int] = []
        for child_joint, parent_joint in getattr(self, "_joint_parent_map", {}).items():
            if (
                child_joint in joint_index_by_name
                and parent_joint in joint_index_by_name
            ):
                joint_to_joint_src.append(joint_index_by_name[parent_joint])
                joint_to_joint_dst.append(joint_index_by_name[child_joint])

        joint_to_foot_src: list[int] = []
        joint_to_foot_dst: list[int] = []
        for foot_joint in foot_names:
            parent_joint = getattr(self, "_joint_parent_map", {}).get(foot_joint)
            if parent_joint in joint_index_by_name and foot_joint in foot_index_by_name:
                joint_to_foot_src.append(joint_index_by_name[parent_joint])
                joint_to_foot_dst.append(foot_index_by_name[foot_joint])

        foot_to_joint_src = list(joint_to_foot_dst)
        foot_to_joint_dst = list(joint_to_foot_src)

        for edge_type, node_pairs in self.spec.node_edge_relations().items():
            if edge_type == "connect":
                for source_node_type, target_node_type in node_pairs:
                    edge_key = (source_node_type, edge_type, target_node_type)

                    if (
                        source_node_type == QHMGNodeType.BASE.value
                        and target_node_type == QHMGNodeType.JOINT.value
                    ):
                        src = [0] * len(root_joint_indices)
                        dst = root_joint_indices
                        edge_index_dict[edge_key] = self._edge_index_from_pairs(
                            src, dst
                        )
                    elif (
                        source_node_type == QHMGNodeType.JOINT.value
                        and target_node_type == QHMGNodeType.BASE.value
                    ):
                        src = root_joint_indices
                        dst = [0] * len(root_joint_indices)
                        edge_index_dict[edge_key] = self._edge_index_from_pairs(
                            src, dst
                        )
                    elif (
                        source_node_type == QHMGNodeType.JOINT.value
                        and target_node_type == QHMGNodeType.JOINT.value
                    ):
                        edge_index_dict[edge_key] = self._edge_index_from_pairs(
                            joint_to_joint_src,
                            joint_to_joint_dst,
                        )
                    elif (
                        source_node_type == QHMGNodeType.JOINT.value
                        and target_node_type == QHMGNodeType.FOOT.value
                    ):
                        edge_index_dict[edge_key] = self._edge_index_from_pairs(
                            joint_to_foot_src,
                            joint_to_foot_dst,
                        )
                    elif (
                        source_node_type == QHMGNodeType.FOOT.value
                        and target_node_type == QHMGNodeType.JOINT.value
                    ):
                        edge_index_dict[edge_key] = self._edge_index_from_pairs(
                            foot_to_joint_src,
                            foot_to_joint_dst,
                        )
                    else:
                        source_indices = node_type_index_dict[source_node_type]
                        target_indices = node_type_index_dict[target_node_type]
                        edge_index_dict[edge_key] = self._pair_to_edge_index(
                            source_indices,
                            target_indices,
                        )

        if self.model_type == ModelType.MS_HGNN:
            for (
                edge_type,
                node_pairs,
            ) in self.spec.node_edge_symmetry_relations().items():
                for source_node_type, target_node_type in node_pairs:
                    edge_key = (source_node_type, edge_type, target_node_type)
                    source_indices = node_type_index_dict[source_node_type]
                    target_indices = node_type_index_dict[target_node_type]

                    group_permutation = (
                        self._resolve_group_permutation_from_spec(
                            source_node_type, edge_type
                        )
                        if source_node_type == target_node_type
                        else None
                    )

                    if (
                        group_permutation is not None
                        and len(group_permutation) > 0
                        and len(source_indices) == len(target_indices)
                        and len(source_indices) % len(group_permutation) == 0
                    ):
                        rows_per_group = len(source_indices) // len(group_permutation)
                        src: list[int] = []
                        dst: list[int] = []
                        for group_idx in range(len(group_permutation)):
                            target_group = group_permutation[group_idx]
                            for offset in range(rows_per_group):
                                src.append(group_idx * rows_per_group + offset)
                                dst.append(target_group * rows_per_group + offset)
                        edge_index_dict[edge_key] = self._edge_index_from_pairs(
                            src, dst
                        )
                    else:
                        edge_index_dict[edge_key] = self._pair_to_edge_index(
                            source_indices,
                            target_indices,
                        )

        return edge_index_dict

    def _resolve_group_permutation_from_spec(
        self,
        node_type: str,
        edge_type: str,
    ) -> list[int] | None:
        """Resolve group permutation without relying on a concrete spec attribute.

        Searches all entries in symmetry_permutation_mapping() and returns the
        first matching permutation for (node_type, edge_type).
        """
        permutation_mapping = self.spec.symmetry_permutation_mapping()
        for per_symmetry in permutation_mapping.values():
            node_mapping = per_symmetry.get(node_type)
            if node_mapping is None:
                continue
            permutation = node_mapping.get(edge_type)
            if permutation is not None and len(permutation) > 0:
                return permutation
        return None

    def _edge_index_from_pairs(self, src: list[int], dst: list[int]) -> torch.Tensor:
        if len(src) == 0 or len(dst) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        if len(src) != len(dst):
            raise ValueError(
                f"Edge source/target length mismatch: {len(src)} != {len(dst)}"
            )
        return torch.tensor([src, dst], dtype=torch.long)

    def _pair_to_edge_index(
        self,
        source_indices: list[int],
        target_indices: list[int],
    ) -> torch.Tensor:
        """
        Build hetero edge index tensor with shape [2, E].

        STRATEGY:
        - If one side has one node: one-to-many
        - If both sides have equal count: pairwise by index
        - Else: dense cartesian product
        """
        if len(source_indices) == 0 or len(target_indices) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        src: list[int] = []
        dst: list[int] = []

        if len(source_indices) == 1:
            src = [source_indices[0]] * len(target_indices)
            dst = list(target_indices)
        elif len(target_indices) == 1:
            src = list(source_indices)
            dst = [target_indices[0]] * len(source_indices)
        elif len(source_indices) == len(target_indices):
            src = list(source_indices)
            dst = list(target_indices)
        else:
            for source_index in source_indices:
                for target_index in target_indices:
                    src.append(source_index)
                    dst.append(target_index)

        return torch.tensor([src, dst], dtype=torch.long)


BaseParser.register(".usd", RobotType.UNITREE_GO2)(UnitreeGO2USDParser)  # type: ignore[arg-type]
