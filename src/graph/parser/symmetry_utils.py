"""Helpers to resolve symmetry row permutations from robot node names."""

from collections.abc import Callable, Sequence


def build_name_permutation(
    names: Sequence[str],
    partner_name: Callable[[str], str],
) -> list[int]:
    """Build ``new_row[i] = old_row[permutation[i]]`` from symmetric name pairs.

    Nodes whose partner name is absent (e.g., central joints such as the waist)
    map to themselves.
    """
    name_to_index = {name: index for index, name in enumerate(names)}
    permutation: list[int] = []
    for name in names:
        partner = partner_name(name)
        permutation.append(name_to_index.get(partner, name_to_index[name]))
    return permutation


def left_right_partner(name: str) -> str:
    """Map a biped node name to its sagittal (left-right) mirror name."""
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


QUADRUPED_LEFT_RIGHT = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
QUADRUPED_FRONT_BACK = {"FL": "RL", "RL": "FL", "FR": "RR", "RR": "FR"}


def quadruped_left_right_partner(name: str) -> str:
    """Map a quadruped node name to its left-right mirror name."""
    prefix = name[:2]
    if prefix in QUADRUPED_LEFT_RIGHT:
        return QUADRUPED_LEFT_RIGHT[prefix] + name[2:]
    return name


def quadruped_front_back_partner(name: str) -> str:
    """Map a quadruped node name to its front-back mirror name."""
    prefix = name[:2]
    if prefix in QUADRUPED_FRONT_BACK:
        return QUADRUPED_FRONT_BACK[prefix] + name[2:]
    return name
