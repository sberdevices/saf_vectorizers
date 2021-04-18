from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from core.text_preprocessing.preprocessing_result import TextPreprocessingResult

from saf_vectorizers.basic_vectorizer import Vectorizer


class USEVectorizer(Vectorizer):

    VECTORIZER_TYPE = "use"
    MODEL_NAME = "universal_sentence_encoder"
    # Размер ембеддинг вектора предобученной модели
    EMBEDDING_SIZE = 512
    # По дефолту модель принимает оригинальный текст как есть, без изменений, лемматизация не нужна
    USE_NORMALIZED_TEXT = False

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        super(USEVectorizer, self).__init__(settings)
        self.use_normalized_text = self.settings.get("use_normalized_text", self.USE_NORMALIZED_TEXT)
        self.session = self._get_session()

    def load_model(self, model_path: str) -> Any:
        return hub.Module(model_path)

    @property
    def size(self) -> int:
        return self.EMBEDDING_SIZE

    @staticmethod
    def _get_session() -> tf.Session:
        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return sess

    def _close_session(self) -> None:
        self.session.close()

    def vectorize(self, text_preprocessing_result: TextPreprocessingResult) -> np.ndarray:
        text = text_preprocessing_result.raw
        text = text["original_text"] if not self.use_normalized_text else text["normalized_text"]
        texts = [text]
        vector = self.session.run(self.model(texts))
        return vector


# TODO: Код оставлен для отладки и демонстрации работоспособности, позже удалить
if __name__ == "__main__":
    import time
    text_pr_result = TextPreprocessingResult(
        {"original_text": "хочу узнать прогноз погоды на завтра в москве",
         "normalized_text": "хотеть узнать прогноз погода на завтра москва ."}
    )
    print("Model initialisation...")
    start = time.time()
    use_vectorizer = USEVectorizer()
    end = time.time()
    print(f"Elapsed time on INITIALISATION is {end - start}")
    print("Start inference...")
    infer_start = time.time()
    emedding_vector = use_vectorizer.vectorize(text_pr_result)
    infer_end = time.time()
    print(f"Elapsed time on INFER is {infer_end - infer_start}")
    print("*** Result: ***")
    print(emedding_vector)
    print(emedding_vector.shape)
    print(use_vectorizer.size)
