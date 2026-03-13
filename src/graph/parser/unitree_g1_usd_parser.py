from pathlib import Path
from typing import Self

from pxr import Usd, UsdPhysics  # type: ignore
import torch

from src.graph.spec.bhmg import BHMGNodeType
from src.graph.morphology.base_morphology import RobotMorphology
from src.graph.spec.base_spec import BaseSpec
from src.config.train_enum import ModelType, RobotType
from src.config.batch_schema import edge_index_dict_type
from src.graph.parser.base_parser import BaseParser


class UnitreeG129DOFUSDParser(BaseParser):
    """
    Parse Unitree G1 robot morphology from USD (Pixar Universal Scene Description) files.

    WORKFLOW:
    1. Load base USD (contains robot hierarchy/geometry)
    2. Load physics USD (contains joint definitions)
    3. Extract joint parents/children from physics stage
    4. Build node list from joints (base, joints, feet, hands)
    5. Build edge connectivity based on joint hierarchy
    6. Create edge_indices tensors for each edge type
    7. Return RobotMorphology

    FILE STRUCTURE:
    - Base USD: g1_29dof_rev_1_0.usd (robot geometry/structure)
    - Physics USD: in configuration/ subfolder with _physics.usd suffix

    NODE IDENTIFICATION:
    - Base: root link (torso/pelvis)
    - Joints: all physics.Joint prims in hierarchy
    - Feet: child links of ankle joints (contains "foot" or "ankle" in name)
    - Hands: child links of wrist joints (contains "hand" or "wrist" in name)
    """

    def __init__(self, model_type: ModelType, spec: BaseSpec, parser_path: Path):
        """
        Initialize USD parser.

        INPUT:
        - model_type: Model type (e.g., MS_HGNN)
        - spec: BHMG specification
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
        - model_type: Model type (e.g., MS_HGNN)
        - spec: BHMG specification
        - parser_path: Path to base USD file

        OUTPUT:
        - Initialized USDParser ready to call parse()

        EXAMPLE:
        parser = USDParser.build_from(spec, "usd/g1_29dof_rev_1_0.usd")
        morphology = parser.parse()
        """
        return cls(model_type, spec, parser_path)

    def find_physics_usd_path(self) -> Path:
        """
        Find physics USD file corresponding to base USD.

        LOGIC:
        - Base: g1_29dof_rev_1_0.usd
        - Physics: configuration/g1_29dof_rev_1_0_physics.usd

        OUTPUT:
        - Path to physics USD file

        RAISES:
        - FileNotFoundError: If physics USD not found
        """
        base_stem = self.parser_path.stem

        # Try configuration/ subfolder first
        physics_path = (
            self.parser_path.parent / "configuration" / f"{base_stem}_physics.usd"
        )
        if physics_path.exists():
            return physics_path

        # Try same directory
        physics_path = self.parser_path.parent / f"{base_stem}_physics.usd"
        if physics_path.exists():
            return physics_path

        # Try removing revision suffix
        base_stem_no_rev = base_stem.split("_rev")[0]
        physics_path = (
            self.parser_path.parent
            / "configuration"
            / f"{base_stem_no_rev}_physics.usd"
        )
        if physics_path.exists():
            return physics_path

        raise FileNotFoundError(
            f"Could not find physics USD for {self.parser_path}\n"
            f"Searched for:\n"
            f"  {self.parser_path.parent / 'configuration' / f'{base_stem}_physics.usd'}\n"
            f"  {self.parser_path.parent / f'{base_stem}_physics.usd'}"
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
            BHMGNodeType.BASE.value: [],
            BHMGNodeType.JOINT.value: [],
            BHMGNodeType.FOOT.value: [],
            BHMGNodeType.HAND.value: [],
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

        for joint_name in self.joint_names:
            parent_joint = joint_parent_map[joint_name]
            children = joint_children_map[joint_name]
            name_lower = joint_name.lower()

            if parent_joint is None and len(children) > 0:
                # G1 root-level articulated joints include a waist/base joint and
                # bilateral hip roots. Keep waist as BASE and treat other articulated
                # roots as JOINT so all locomotion DOFs are represented.
                if "waist" in name_lower:
                    node_type[BHMGNodeType.BASE.value].append(joint_name)
                else:
                    node_type[BHMGNodeType.JOINT.value].append(joint_name)
            elif parent_joint is not None and len(children) > 0:
                node_type[BHMGNodeType.JOINT.value].append(joint_name)
            elif parent_joint is not None and len(children) == 0:
                if "foot" in name_lower or "ankle" in name_lower:
                    node_type[BHMGNodeType.FOOT.value].append(joint_name)
                elif "hand" in name_lower or "wrist" in name_lower:
                    node_type[BHMGNodeType.HAND.value].append(joint_name)
                else:
                    raise ValueError(
                        f"Unclassified leaf joint: {joint_name}\n"
                        f"Parent joint: {parent_joint}\n"
                        f"Children: {children}"
                    )
            else:
                raise ValueError(
                    f"Unclassified leaf joint: {joint_name}\n"
                    f"Parent joint: {parent_joint}\n"
                    f"Children: {children}"
                )

        return node_type

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

        for edge_type, node_pairs in self.spec.node_edge_relations().items():
            if edge_type == "connect":
                for source_node_type, target_node_type in node_pairs:
                    edge_key = (source_node_type, edge_type, target_node_type)
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


class UnitreeG123DOFUSDParser(UnitreeG129DOFUSDParser):
    """Parser for Unitree G1 23-DOF robot."""

    pass


BaseParser.register(".usd", RobotType.UNITREE_G1_29DOF)(UnitreeG129DOFUSDParser)  # type: ignore[arg-type]
BaseParser.register(".usd", RobotType.UNITREE_G1_23DOF)(UnitreeG123DOFUSDParser)  # type: ignore[arg-type]
