"""Load the studio child graphs the chatbot orchestrates.

Loading is eager, exactly as in the original module, so a broken child graph
fails at import time rather than mid-conversation.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

from app.config.const.chatbot import (
    CV_GRAPH_PATH,
    CV_REVIEW_GRAPH_PATH,
    JOB_GRAPH_PATH,
    MATCHING_SCORE_GRAPH_PATH,
    MODULE_CV_EXTRACTION,
    MODULE_CV_REVIEW,
    MODULE_JOB_EXTRACTION,
    MODULE_MATCHING_SCORE,
)
from app.models.chat_model import ChatModel


class SubgraphRepository:
    """Access to the CV extraction, job extraction, matching, and review graphs."""

    def __init__(self) -> None:
        self._cv_module: ModuleType = self._load_graph(
            CV_GRAPH_PATH,
            MODULE_CV_EXTRACTION,
        )
        self._job_module: ModuleType = self._load_graph(
            JOB_GRAPH_PATH,
            MODULE_JOB_EXTRACTION,
        )
        self._matching_score_module: ModuleType = self._load_graph(
            MATCHING_SCORE_GRAPH_PATH,
            MODULE_MATCHING_SCORE,
        )
        self._cv_review_module: ModuleType = self._load_graph(
            CV_REVIEW_GRAPH_PATH,
            MODULE_CV_REVIEW,
        )

    @staticmethod
    def _load_graph(path: Path, module_name: str) -> ModuleType:
        spec: ModuleSpec | None = importlib.util.spec_from_file_location(
            module_name,
            path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load child graph module from {path}")

        module: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "graph"):
            raise ImportError(f"Child graph module does not export graph: {path}")

        return module

    @property
    def cv_extraction(self) -> Any:
        return self._cv_module.graph

    @property
    def job_extraction(self) -> Any:
        return self._job_module.graph

    @property
    def matching_score(self) -> Any:
        return self._matching_score_module.graph

    def build_cv_review_graph(self, *, chat_model: ChatModel | None = None) -> Any:
        return self._cv_review_module.build_graph(chat_model=chat_model)

    def classify_fit_verdict(self, **kwargs: Any) -> tuple[Any, Any]:
        return self._matching_score_module.classify_fit_verdict(**kwargs)


SUBGRAPH_REPOSITORY: SubgraphRepository = SubgraphRepository()
