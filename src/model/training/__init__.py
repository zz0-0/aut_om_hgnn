"""Training package initialization.

Imports concrete Lightning models for registration side effects.
"""

from src.model.training import mi_hgnn_lit_model  # type: ignore # noqa: F401
from src.model.training import ms_hgnn_lit_model  # type: ignore # noqa: F401
