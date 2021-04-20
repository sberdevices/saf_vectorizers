import os
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf
from core.text_preprocessing.preprocessing_result import TextPreprocessingResult

from saf_vectorizers.basic_vectorizer import Vectorizer, PRETRAINED_MODELS_PATH
from saf_vectorizers.preprocessing import FullTokenizer
from saf_vectorizers.utils import compute_input_array


class SBERTVectorizer(Vectorizer):
    """SBERT (SentenceBERT) векторизатор."""

    VECTORIZER_TYPE = "sbert"
    MODEL_NAME = "sbert/sbert.graphdef"
    # Размер ембеддинг вектора предобученной модели
    EMBEDDING_SIZE = 1024
    # Количество сегментов т.е. максимальная длина обрабатываемой последовательности (24 токена),
    # это параметр был использован при обучении SBERT, является контстантой
    MAX_SEQ_LEN = 24
    SBERT_VOCAB_PATH = os.path.join(PRETRAINED_MODELS_PATH, "sbert/sbert_vocab.txt")
    OUTPUT_TO_USE = "sentence"

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        super(SBERTVectorizer, self).__init__(settings)
        self.output_to_use = self.settings.get("output_to_use", self.OUTPUT_TO_USE)
        self.restored_graph, self.tf_session = self.model
        self.tokenizer = FullTokenizer(vocab_file=self.SBERT_VOCAB_PATH, do_lower_case=False)
        # Первый запуск BERT'a всегда долгий поэтому выполняется в конструкторе
        self.x_id, self.x_mask, self.x_seg, self.y1, self.y2 = self._restore_model(self.restored_graph)
        _ = self._predict([[0] * self.MAX_SEQ_LEN], [[0] * self.MAX_SEQ_LEN], [[0] * self.MAX_SEQ_LEN])

    @staticmethod
    def _load_graph(frozen_graph_filename: str) -> tf.Graph:
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        return graph

    @staticmethod
    def _restore_model(restored_graph: tf.Graph):
        graph_ops = restored_graph.get_operations()
        input_op = [graph_ops[0].name, graph_ops[1].name, graph_ops[2].name]
        output_op = [graph_ops[-2].name, graph_ops[-1].name]
        [x_id, x_mask, x_seg] = [restored_graph.get_tensor_by_name(input_op[i] + ":0") for i, k in enumerate(input_op)]
        y1 = restored_graph.get_tensor_by_name(output_op[0] + ":0")
        y2 = restored_graph.get_tensor_by_name(output_op[1] + ":0")
        return x_id, x_mask, x_seg, y1, y2

    def _predict(self, in_ids, in_masks, in_segs) -> np.ndarray:
        return self.tf_session.run(
            [self.y1, self.y2],
            feed_dict={self.x_id: in_ids, self.x_mask: in_masks, self.x_seg: in_segs}
        )

    def load_model(self, model_path: str) -> Any:
        restored_graph = self._load_graph(model_path)
        tf_session = tf.Session(graph=restored_graph)
        return restored_graph, tf_session

    def vectorize(self, text_preprocessing_result: TextPreprocessingResult) -> np.ndarray:
        text = text_preprocessing_result.raw
        text = text["original_text"] if not self.use_normalized_text else text["normalized_text"]
        texts = [text]
        in_ids, in_masks, in_seqs = compute_input_array(texts, self.tokenizer, self.MAX_SEQ_LEN)
        outs = self._predict(in_ids, in_masks, in_seqs)
        vector = outs[0] if self.output_to_use == "sentence" else outs[1]
        return vector

    @property
    def size(self) -> int:
        return self.EMBEDDING_SIZE
