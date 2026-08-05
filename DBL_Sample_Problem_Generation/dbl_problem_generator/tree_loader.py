"""Interfaces for loading DBL trees into a graph database."""

from __future__ import annotations

from typing import Any


class DblTreeLoader:
    """Project-specific loader for persisting DBL tree structures."""

    def load_tree(self, tree_payload: dict[str, Any]) -> None:
        """
        Load a single DBL tree into the backing graph database.

        Replace this method body with your existing ingestion logic.
        """
        raise NotImplementedError("Implement DBL tree loading logic in this class.")
