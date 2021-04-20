"""
Файл содержит набор полезных функций, которые могут быть переиспользованы для препроцессинга текстов,
а также при добавление новых векторизаторов.
"""

import collections
import unicodedata
from typing import Union, List, Dict, Any

import numpy as np
import tensorflow as tf


def convert_to_unicode(text: Union[str, bytes]) -> str:
    if isinstance(text, str):
        result = text
    elif isinstance(text, bytes):
        result = text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    return result


def is_whitespace(char: str) -> bool:
    # \t, \n, и \r технически являются "contorl characters",
    # но в рамках препроцессинга текстов их следует рассматривать по аналогии с пробелами
    result = False
    if char in (" ", "\t", "\n", "\r"):
        result = True
    cat = unicodedata.category(char)
    if cat == "Zs":
        result = True
    return result


def is_control(char: str) -> bool:
    result = False
    if char in ("\t", "\n", "\r"):
        result = True
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        result = True
    return result


def is_punctuation(char: str) -> bool:
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for consistency.
    result = False
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        result = True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        result = True
    return result


def whitespace_tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        tokens = []
    else:
        tokens = text.split()
    return tokens


def load_vocab(vocab_file: str) -> Dict[str, int]:
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab: Union[Dict[str, int], Dict[int, str]], items: Union[str, int]) -> List[Union[str, int]]:
    return [vocab[item] for item in items]


def get_masks(tokens: List[str], max_seq_length: int) -> List[int]:
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens: List[str], max_seq_length: int) -> List[int]:
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [0] * len(tokens) + [0] * (max_seq_length - len(tokens))


def trim_input(text: str, tokenizer: Any, max_seq_length: int) -> List[Any]:
    result = tokenizer.tokenize(text)
    if len(result) > max_seq_length - 2:
        result = result[0: (max_seq_length - 2)]
    return result


def get_ids(tokens: List[Any], tokenizer: Any, max_seq_length: int) -> List[int]:
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def convert_to_bert_inputs(text: List[Any], tokenizer: Any, max_seq_length: int) -> List[List]:
    stoken = ["[CLS]"] + text + ["[SEP]"]
    input_ids = get_ids(tokens=stoken, tokenizer=tokenizer, max_seq_length=max_seq_length)
    input_masks = get_masks(tokens=stoken, max_seq_length=max_seq_length)
    input_segments = get_segments(tokens=stoken, max_seq_length=max_seq_length)
    return [input_ids, input_masks, input_segments]


def compute_input_array(texts: List[str], tokenizer: Any, max_seq_length: int) -> List[np.array]:
    input_ids, input_masks, input_segments = [], [], []
    for text in texts:
        t = trim_input(text, tokenizer, max_seq_length)
        ids, masks, segments = convert_to_bert_inputs(t, tokenizer, max_seq_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [np.array(input_ids, dtype=np.int32), np.array(input_masks, dtype=np.int32),
            np.array(input_segments, dtype=np.int32)]
