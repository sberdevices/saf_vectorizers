import collections
import unicodedata

import numpy as np
import tensorflow as tf


def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def is_whitespace(char):
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def load_vocab(vocab_file):
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


def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def get_masks(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [0] * len(tokens) + [0] * (max_seq_length - len(tokens))


def trim_input(text, tokenizer, max_seq_length):
    t = tokenizer.tokenize(text)
    if len(t) > max_seq_length - 2:
        t = t[0: (max_seq_length - 2)]
    return t


def get_ids(tokens, tokenizer, max_seq_length):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def convert_to_bert_inputs(text, tokenizer, max_seq_length):
    stoken = ["[CLS]"] + text + ["[SEP]"]
    input_ids = get_ids(tokens=stoken, tokenizer=tokenizer, max_seq_length=max_seq_length)
    input_masks = get_masks(tokens=stoken, max_seq_length=max_seq_length)
    input_segments = get_segments(tokens=stoken, max_seq_length=max_seq_length)
    return [input_ids, input_masks, input_segments]


def compute_input_array(texts, tokenizer, max_seq_length):
    input_ids, input_masks, input_segments = [], [], []
    for text in texts:
        t = text
        t = trim_input(t, tokenizer, max_seq_length)
        ids, masks, segments = convert_to_bert_inputs(t, tokenizer, max_seq_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [np.array(input_ids, dtype=np.int32), np.array(input_masks, dtype=np.int32),
            np.array(input_segments, dtype=np.int32)]
