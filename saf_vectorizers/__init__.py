from core.basic_models.classifiers.vectorizer_models import vectorizers as saf_vectorizers

from saf_vectorizers.basic_vectorizer import Vectorizer
from saf_vectorizers.fasttext_vect import FastTextVectorizer
from saf_vectorizers.sbert import SBERTVectorizer
from saf_vectorizers.universal_sentence_encoder import USEVectorizer
from saf_vectorizers.word2vec_vect import Word2VecVectorizer

# Словарь содержит все типы (ключи) всех классов-наследников (значения) базового класса Vectorizer
VECTORIZERS = {cls.VECTORIZER_TYPE: cls for cls in Vectorizer.__subclasses__()}


def vectorizer_factory(vectorizer_type: str, *args, **kwargs) -> Vectorizer:
    try:
        vectorizer_class = VECTORIZERS[vectorizer_type]
    except KeyError:
        raise Exception(f"Unknown vectorizer type: {vectorizer_type}! It should be one of: {VECTORIZERS.keys()} types!")
    return vectorizer_class()


def on_startup(app_config, manager) -> None:
    """Функция вызывается в smart_app_framework в методе activate_plugins в момент инициализации смартаппа,
    и выполняет регистрацию и загрузку всех векторизаторов.
    """
    saf_vectorizers["fasttext"] = FastTextVectorizer()
    saf_vectorizers["sbert"] = SBERTVectorizer()
    saf_vectorizers["use"] = USEVectorizer()
    saf_vectorizers["word2vec"] = Word2VecVectorizer()


__all__ = [
    "FastTextVectorizer",
    "SBERTVectorizer",
    "VECTORIZERS",
    "Vectorizer",
    "vectorizer_factory",
    "USEVectorizer",
    "Word2VecVectorizer",
    "on_startup"
]
