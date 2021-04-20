from typing import Dict, Any, Optional

import fasttext
import numpy as np
from core.text_preprocessing.preprocessing_result import TextPreprocessingResult

from saf_vectorizers.basic_vectorizer import Vectorizer


class FastTextVectorizer(Vectorizer):
    """FastText векторизатор."""

    VECTORIZER_TYPE = "fasttext"
    MODEL_NAME = "fasttext/fasttext.bin"

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        super(FastTextVectorizer, self).__init__(settings)

    def load_model(self, model_path: str) -> Any:
        model = fasttext.load_model(model_path)
        return model

    @property
    def size(self) -> int:
        return self.model.get_dimension()

    def vectorize(self, text_preprocessing_result: TextPreprocessingResult) -> np.ndarray:
        text = text_preprocessing_result.raw
        text = text["original_text"] if not self.use_normalized_text else text["normalized_text"]
        vector = self.model.get_sentence_vector(text)
        return vector
