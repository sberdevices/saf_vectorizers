import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np
from core.text_preprocessing.preprocessing_result import TextPreprocessingResult

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

PRETRAINED_MODELS_PATH = os.path.join(PROJECT_ROOT_PATH, "static")


class Vectorizer(ABC):
    """Базовый класс для сущности Векторизатор."""

    VECTORIZER_TYPE = None
    MODEL_NAME = None
    # По дефолту все нейросетевые модели работают с оригинальным текстом как есть, лемматизация не осуществлялась
    USE_NORMALIZED_TEXT = False

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self.settings = settings if settings else {}
        vectorizer_type = self.settings.get("type")
        if vectorizer_type:
            self._check_vectorizer_type(vectorizer_type)
        self.use_normalized_text = self.settings.get("use_normalized_text", self.USE_NORMALIZED_TEXT)
        self.model = self.load_model(os.path.join(PRETRAINED_MODELS_PATH, self.MODEL_NAME))

    def _check_vectorizer_type(self, vectorizer_type: str) -> None:
        if vectorizer_type != self.VECTORIZER_TYPE:
            raise Exception(f"Inappropriate vectorizer type: {vectorizer_type}, it should be {self.VECTORIZER_TYPE}")

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Метод осуществляет загрузку предобученной модели."""
        raise NotImplementedError

    @abstractmethod
    def vectorize(self, text_preprocessing_result: TextPreprocessingResult) -> np.ndarray:
        """Метод осуществляет векторизацию текстовой реплики."""
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        """Метод возвращает размер embedding'а, т.е длину вектора."""
        raise NotImplementedError
