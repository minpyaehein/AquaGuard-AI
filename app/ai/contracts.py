from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

@dataclass
class PredictionContext:
    country_code: str
    observed_at: str
    source_times: dict[str, str] = field(default_factory=dict)
    source_quality: dict[str, float] = field(default_factory=dict)

@dataclass
class ModelOutput:
    probability: np.ndarray
    confidence: np.ndarray
    model_name: str
    model_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

class ModelPlugin(ABC):
    task: str
    @abstractmethod
    def load(self, path: Path, device: str = "cpu") -> None: ...
    @abstractmethod
    def predict(self, inputs: Any, context: PredictionContext) -> ModelOutput: ...
    @abstractmethod
    def health(self) -> dict[str, Any]: ...
