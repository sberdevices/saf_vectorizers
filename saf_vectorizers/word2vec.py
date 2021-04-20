from typing import Dict, Any, Optional

import numpy as np
from core.text_preprocessing.preprocessing_result import TextPreprocessingResult
from gensim.models import KeyedVectors

from saf_vectorizers.basic_vectorizer import Vectorizer


class Word2VecVectorizer(Vectorizer):

    VECTORIZER_TYPE = "word2vec"
    MODEL_NAME = "word2vec/model.bin"
    BINARY_MOOD = True

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        super(Word2VecVectorizer, self).__init__(settings)

    def load_model(self, model_path: str) -> Any:
        model = KeyedVectors.load_word2vec_format(model_path, binary=self.BINARY_MOOD)
        return model

    @property
    def size(self) -> int:
        return self.model.vector_size

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
