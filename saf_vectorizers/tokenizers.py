"""Файл содержит токенизаторы, которые могут быть переиспользованы для препроцессинга текстов,
а также при добавление новых векторизаторов."""

import unicodedata
from typing import Optional, List, Dict

from saf_vectorizers.utils import load_vocab, convert_by_vocab, convert_to_unicode, whitespace_tokenize, \
    is_punctuation, is_control, is_whitespace


class Tokenizer:
    """Базовый класс для сущности Токенизатор."""

    def __init__(self, do_lower_case: Optional[bool] = True) -> None:
        self.do_lower_case = do_lower_case

    def tokenize(self, text: str) -> List[str]:
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []

        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    @staticmethod
    def _run_strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    @staticmethod
    def _run_split_on_punc(text: str) -> List[str]:
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text: str) -> str:
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _is_chinese_char(cp: int) -> bool:
        result = False
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
                or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF
                or 0x2F800 <= cp <= 0x2FA1F):
            result = True
        return result

    @staticmethod
    def _clean_text(text: str) -> str:
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer:
    """Запускает 'word-piece' токенизацию."""

    def __init__(self, vocab: Dict[str, int], unk_token: Optional[str] = "[UNK]",
                 max_input_chars_per_word: Optional[int] = 100) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> List[str]:
        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class FullTokenizer:
    """Запускает 'end-to-end' токенизацию."""

    def __init__(self, vocab_file: str, do_lower_case: Optional[bool] = True) -> None:
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = Tokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text: List[str]) -> List[str]:
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return convert_by_vocab(self.inv_vocab, ids)

    def mark_unk_tokens(self, tokens: List[str], unk_token: Optional[str] = "[UNK]") -> List[str]:
        return [t if t in self.vocab else unk_token for t in tokens]
