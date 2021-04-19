from saf_vectorizers.basic_vectorizer import Vectorizer
from saf_vectorizers.sbert import SBERTVectorizer
from saf_vectorizers.universal_sentence_encoder import USEVectorizer
from saf_vectorizers.word2vec import Word2VecVectorizer


VECTORIZERS = {cls.VECTORIZER_TYPE: cls for cls in Vectorizer.__subclasses__()}


def vectorizer_factory(vectorizer_type: str, *args, **kwargs) -> Vectorizer:
    try:
        vectorizer_class = VECTORIZERS[vectorizer_type]
    except KeyError:
        raise Exception(f"Unknown vectorizer type: {vectorizer_type}! It should be one of: {VECTORIZERS.keys()} types!")
    return vectorizer_class
