"""
Данный скрипт находится здесь лишь для быстрой локальной проверки и демонстрации работоспособности всех векторизаторов,
и никак не используется в smart_app_framework.
Скрипт оставлен для наглядности и поможет замерить время загрузки и инференса каждого из векторизаторов.
"""

import time

from core.text_preprocessing.preprocessing_result import TextPreprocessingResult

from saf_vectorizers import vectorizer_factory, VECTORIZERS

# Пример объекта, который поступает на вход в метод vectorize
PREPROCESSED_TEXT = TextPreprocessingResult(
        {
            "original_text": "хочу узнать прогноз погоды на завтра в москве",
            "normalized_text": "хотеть узнать прогноз погода на завтра в москва .",
            "tokenized_elements_list": [
                {"text": "хочу", "grammem_info": {
                    "aspect": "impf", "mood": "ind", "number": "sing", "person": "1", "tense": "notpast",
                    "transitivity": "tran", "verbform": "fin", "voice": "act", "raw_gram_info":
                    "aspect=impf|mood=ind|number=sing|person=1|tense=notpast|transitivity=tran|verbform=fin|voice=act",
                    "part_of_speech": "VERB"}, "lemma": "хотеть"},
                {"text": "узнать", "grammem_info": {"aspect": "perf", "transitivity": "tran", "verbform": "inf",
                 "raw_gram_info": "aspect=perf|transitivity=tran|verbform=inf", "part_of_speech": "VERB"},
                 "lemma": "узнать"},
                {"text": "прогноз", "grammem_info": {"animacy": "inan", "case": "acc", "gender": "masc",
                                                     "number": "sing", "raw_gram_info":
                                                         "animacy=inan|case=acc|gender=masc|number=sing",
                                                     "part_of_speech": "NOUN"}, "lemma": "прогноз"},
                {"text": "погоды", "grammem_info": {"animacy": "inan", "case": "gen", "gender": "fem", "number": "sing",
                                                    "raw_gram_info": "animacy=inan|case=gen|gender=fem|number=sing",
                                                    "part_of_speech": "NOUN"}, "lemma": "погода"},
                {"text": "на", "grammem_info": {"raw_gram_info": "", "part_of_speech": "ADP"}, "lemma": "на"},
                {"text": "завтра",
                 "grammem_info": {"degree": "pos", "raw_gram_info": "degree=pos", "part_of_speech": "ADV"},
                 "lemma": "завтра"},
                {"text": "в", "grammem_info": {"raw_gram_info": "", "part_of_speech": "ADP"}, "lemma": "в"},
                {"text": "москве", "grammem_info": {"animacy": "inan", "case": "loc", "gender": "fem", "number": "sing",
                                                    "raw_gram_info": "animacy=inan|case=loc|gender=fem|number=sing",
                                                    "part_of_speech": "NOUN"}, "lemma": "москва"},
                {"text": ".", "lemma": ".", "token_type": "SENTENCE_ENDPOINT_TOKEN", "token_value": {"value": "."},
                 "list_of_token_types_data": [{"token_type": "SENTENCE_ENDPOINT_TOKEN", "token_value": {"value": "."}}]}
            ]
        }
)


def infer_vectorizer(vectorizer_type: str) -> None:
    print(f"******* INFER {vectorizer_type} vectorizer *******")
    print("## Model initialisation...")
    start = time.time()
    vectorizer = vectorizer_factory(vectorizer_type)
    end = time.time()
    print(f"## Elapsed time on INITIALISATION is {end - start}")
    print("## Start inference...")
    infer_start = time.time()
    emedding_vector = vectorizer.vectorize(PREPROCESSED_TEXT)
    infer_end = time.time()
    print(f"## Elapsed time on INFER is {infer_end - infer_start}")
    print("*** Result: ***")
    print(emedding_vector)
    print(emedding_vector.shape)
    print(vectorizer.size)


if __name__ == "__main__":
    for vect_type in VECTORIZERS.keys():
        infer_vectorizer(vect_type)
