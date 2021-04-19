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


# TODO: Код оставлен для отладки и демонстрации работоспособности, позже удалить
if __name__ == "__main__":
    import time
    text_pr_result = TextPreprocessingResult(
        {
            'original_text': 'хочу узнать прогноз погоды на завтра в москве',
            'normalized_text': 'хотеть узнать прогноз погода на завтра в москва .',
            'tokenized_elements_list': [
                {'text': 'хочу', 'grammem_info': {
                    'aspect': 'impf', 'mood': 'ind', 'number': 'sing', 'person': '1', 'tense': 'notpast',
                    'transitivity': 'tran', 'verbform': 'fin', 'voice': 'act', 'raw_gram_info':
                    'aspect=impf|mood=ind|number=sing|person=1|tense=notpast|transitivity=tran|verbform=fin|voice=act',
                    'part_of_speech': 'VERB'}, 'lemma': 'хотеть'},
                {'text': 'узнать', 'grammem_info': {'aspect': 'perf', 'transitivity': 'tran', 'verbform': 'inf',
                 'raw_gram_info': 'aspect=perf|transitivity=tran|verbform=inf', 'part_of_speech': 'VERB'},
                 'lemma': 'узнать'},
                {'text': 'прогноз', 'grammem_info': {'animacy': 'inan', 'case': 'acc', 'gender': 'masc',
                                                     'number': 'sing', 'raw_gram_info':
                                                         'animacy=inan|case=acc|gender=masc|number=sing',
                                                     'part_of_speech': 'NOUN'}, 'lemma': 'прогноз'},
                {'text': 'погоды', 'grammem_info': {'animacy': 'inan', 'case': 'gen', 'gender': 'fem', 'number': 'sing',
                                                    'raw_gram_info': 'animacy=inan|case=gen|gender=fem|number=sing',
                                                    'part_of_speech': 'NOUN'}, 'lemma': 'погода'},
                {'text': 'на', 'grammem_info': {'raw_gram_info': '', 'part_of_speech': 'ADP'}, 'lemma': 'на'},
                {'text': 'завтра',
                 'grammem_info': {'degree': 'pos', 'raw_gram_info': 'degree=pos', 'part_of_speech': 'ADV'},
                 'lemma': 'завтра'},
                {'text': 'в', 'grammem_info': {'raw_gram_info': '', 'part_of_speech': 'ADP'}, 'lemma': 'в'},
                {'text': 'москве', 'grammem_info': {'animacy': 'inan', 'case': 'loc', 'gender': 'fem', 'number': 'sing',
                                                    'raw_gram_info': 'animacy=inan|case=loc|gender=fem|number=sing',
                                                    'part_of_speech': 'NOUN'}, 'lemma': 'москва'},
                {'text': '.', 'lemma': '.', 'token_type': 'SENTENCE_ENDPOINT_TOKEN', 'token_value': {'value': '.'},
                 'list_of_token_types_data': [{'token_type': 'SENTENCE_ENDPOINT_TOKEN', 'token_value': {'value': '.'}}]}
            ]
        }
        )
    print("Model initialisation...")
    start = time.time()
    w2v_vectorizer = Word2VecVectorizer()
    end = time.time()
    print(f"Elapsed time on INITIALISATION is {end - start}")
    print("Start inference...")
    infer_start = time.time()
    emedding_vector = w2v_vectorizer.vectorize(text_pr_result)
    infer_end = time.time()
    print(f"Elapsed time on INFER is {infer_end - infer_start}")
    print("*** Result: ***")
    print(emedding_vector)
    print(emedding_vector.shape)
    print(w2v_vectorizer.size)
