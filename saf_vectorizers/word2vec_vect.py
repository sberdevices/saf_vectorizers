from typing import Dict, Any, Optional

import numpy as np
import word2vec
from core.text_preprocessing.preprocessing_result import TextPreprocessingResult

from saf_vectorizers.basic_vectorizer import Vectorizer


class Word2VecVectorizer(Vectorizer):
    """Word2Vec векторизатор."""

    VECTORIZER_TYPE = "word2vec"
    MODEL_NAME = "word2vec/model.txt"
    # Размер ембеддинг вектора предобученной модели
    EMBEDDING_SIZE = 300

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        super(Word2VecVectorizer, self).__init__(settings)

    def load_model(self, model_path: str) -> Any:
        model = word2vec.load(model_path)
        return model

    @property
    def size(self) -> int:
        return self.EMBEDDING_SIZE

    def vectorize(self, text_preprocessing_result: TextPreprocessingResult) -> np.ndarray:
        sentence_vec = []
        text = text_preprocessing_result.raw["tokenized_elements_list"]
        for token in text:
            if token.get("token_type") and token["token_type"] == "SENTENCE_ENDPOINT_TOKEN":
                break
            token = f"{token['lemma']}_{token['grammem_info']['part_of_speech']}"
            try:
                word_vec = self.model.get_vector(token)
                sentence_vec.append(word_vec)
            except KeyError:
                pass
        return np.mean(np.array(sentence_vec), axis=0)
