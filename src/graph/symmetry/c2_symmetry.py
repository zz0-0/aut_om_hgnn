from typing import Self

import torch

from src.config.batch_schema import HeteroDataBatch, x_dict_type
from src.config.train_enum import SymmetryType
from src.graph.feature.base_feature import BaseFeatureType
from src.graph.spec.base_spec import symmetry_permutation_dict_type
from src.graph.symmetry.base_symmetry import BaseSymmetry, symmetry_edge_dict_type


class C2Symmetry(BaseSymmetry):
    def __init__(
        self,
        symmetry_edge_dict: symmetry_edge_dict_type,
        symmetry_permutation_dict: symmetry_permutation_dict_type | None = None,
    ):
        """Initialize C2 symmetry."""
        super().__init__()
        self.symmetry_edge_dict = symmetry_edge_dict
        self.symmetry_edge_types = symmetry_edge_dict[SymmetryType.C2.value]
        self.symmetry_permutation = (
            symmetry_permutation_dict.get(SymmetryType.C2.value, {})
            if symmetry_permutation_dict is not None
            else {}
        )
        self.combination = self.generate_combination(self.symmetry_edge_types)
        self.reflection_coefficients = self.generate_reflection_coefficients(
            self.symmetry_edge_types
        )

    @classmethod
    def build_from(
        cls,
        symmetry_edge_dict: symmetry_edge_dict_type,
        symmetry_permutation_dict: symmetry_permutation_dict_type | None = None,
    ) -> Self:
        """Build C2 symmetry instance from configuration."""
        if SymmetryType.C2.value not in symmetry_edge_dict.keys():
            raise ValueError(
                f"The symmetry type does not match with configuration. Expected key: {SymmetryType.C2.value}"
            )
        return cls(symmetry_edge_dict, symmetry_permutation_dict)

    def expand_data(self, data_list: list[HeteroDataBatch]) -> list[HeteroDataBatch]:
        """
        Expand input data list according to C2 symmetry.

        INPUT:
        - data_list: list of input data items containing node features, edge features, etc.

        OUTPUT:
        - expanded_data_list: new list where each data item has been duplicated and transformed
          according to C2 symmetry.
        """
        expanded_data_list: list[HeteroDataBatch] = []
        for data in data_list:
            new_data = self._apply_c2_transform(data)  # Apply C2 transformations
            expanded_data_list.append(
                new_data
            )  # Add the transformed data (C2 symmetry)

        return expanded_data_list

    def _apply_c2_transform(self, data: HeteroDataBatch) -> HeteroDataBatch:
        """
        Apply C2 symmetry transformations to the input data in-place.

        This method should implement the specific transformations required to achieve
        C2 symmetry (e.g., swapping left/right limbs, applying appropriate rotations,
        etc.). The exact transformations will depend on how the data is structured and
        what features are present.

        INPUT:
        - data: input data item to be transformed according to C2 symmetry

        OUTPUT:
        - None (the input data is modified in-place)
        """
        new_data = data.clone()  # Create a copy of the input data to modify
        serialized_layout = getattr(data, "feature_type_layout", None)
        new_data.x_dict = self._modify_x_dict(data.x_dict, serialized_layout)

        if "gt" in self.symmetry_edge_types:
            symmetry_op = "gt"
        elif "gs" in self.symmetry_edge_types:
            symmetry_op = "gs"
        else:
            symmetry_op = next(iter(self.symmetry_edge_types.keys()))
        combo = (symmetry_op,)

        if hasattr(new_data, "y_contact"):
            new_data.y_contact = self.apply_output_permutation(
                new_data.y_contact, "foot", combo
            )
        if hasattr(new_data, "y_ground_reaction_force"):
            new_data.y_ground_reaction_force = self.apply_output_permutation(
                new_data.y_ground_reaction_force, "foot", combo
            )
        return new_data

    def _modify_x_dict(
        self,
        x_dict: x_dict_type,
        serialized_layout: dict[str, list[tuple[int, int, str]]] | None,
    ) -> x_dict_type:
        """
        Modify node features in x_dict according to C2 symmetry.

        This method should implement the specific modifications to node features required
        for C2 symmetry (e.g., swapping features of left/right limbs, applying sign
        changes, etc.). The exact modifications will depend on the structure of the
        node features and how they relate to the symmetry.

        INPUT:
        - x_dict: dictionary of node features to be modified in-place

        OUTPUT:
        - Modified x_dict with C2 symmetry applied
        """
        modified_x_dict: x_dict_type = {}
        parsed_layout = self.parse_feature_type_layout(serialized_layout)
        if not parsed_layout:
            parsed_layout = self.infer_feature_type_layout(x_dict)

        if "gt" in self.symmetry_edge_types:
            symmetry_op = "gt"
        elif "gs" in self.symmetry_edge_types:
            symmetry_op = "gs"
        else:
            symmetry_op = next(iter(self.symmetry_edge_types.keys()))

        axes_to_flip = self.symmetry_edge_types[symmetry_op]

        vector_coeff = torch.tensor(
            self._coefficients_from_axes(axes_to_flip, BaseFeatureType.VECTOR_3D),
            dtype=torch.float32,
        )
        pseudo_coeff = torch.tensor(
            self._coefficients_from_axes(
                axes_to_flip,
                BaseFeatureType.PSEUDOVECTOR_3D,
            ),
            dtype=torch.float32,
        )

        for node_type, x in x_dict.items():
            blocks = parsed_layout.get(node_type)
            if blocks is None:
                blocks = self.infer_feature_type_layout({node_type: x}).get(
                    node_type, []
                )

            transformed = self.apply_row_permutation_combo(
                x.clone(), node_type, (symmetry_op,)
            )
            for start, end, feature_type in blocks:
                if end <= start:
                    continue

                if feature_type == BaseFeatureType.SCALAR:
                    continue

                coeff = (
                    vector_coeff
                    if feature_type == BaseFeatureType.VECTOR_3D
                    else pseudo_coeff
                )
                coeff = coeff.to(device=x.device, dtype=x.dtype)

                width = end - start
                if width % 3 != 0:
                    continue
                for offset in range(start, end, 3):
                    transformed[:, offset : offset + 3] = (
                        transformed[:, offset : offset + 3] * coeff
                    )

            if node_type == "base":
                modified_x_dict[node_type] = torch.cat([x, transformed], dim=0)
            else:
                modified_x_dict[node_type] = transformed

        return modified_x_dict


BaseSymmetry.register(SymmetryType.C2)(C2Symmetry)  # type: ignore[arg-type]
