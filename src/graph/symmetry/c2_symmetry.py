from typing import Self

from src.config.batch_schema import HeteroDataBatch
from src.config.train_enum import SymmetryType
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
        - expanded_data_list: new list where each data item has been transformed by a
          sampled C2 group element (identity or left-right reflection).
        """
        return [
            self.apply_symmetry_transform(data, self.sample_combo())
            for data in data_list
        ]


BaseSymmetry.register(SymmetryType.C2)(C2Symmetry)  # type: ignore[arg-type]
