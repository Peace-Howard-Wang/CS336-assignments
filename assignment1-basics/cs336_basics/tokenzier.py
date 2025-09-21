import collections
import pickle
from typing import Iterable

import numpy as np
import regex as re
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.bep_rank = {pair: rank for rank, pair in enumerate(merges)}
        self._byte_cache = [bytes([i]) for i in range(256)]
        self._pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self._split_pattern = None
        if self.special_tokens is not None:
            self._split_pattern = re.compile("(" + "|".join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True)) + ")")
        self.token_to_id = {token: id for id, token in vocab.items()}
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _get_pairs(self, word):
        pair = set()
        for i in range(len(word) - 1):
            pair.add((word[i], word[i+1]))
        return pair

    def bpe_on_token_bytes(self, token_bytes):
        word = [self._byte_cache[b] for b in token_bytes]
        if len(word) == 0:
            return []
        pairs = self._get_pairs(word)
        if pairs is None:
            return word
        while True:
            min_pair = None
            min_rank = None
            for p in pairs:
                if p in self.bep_rank:
                    r = self.bep_rank[p]
                    if min_rank is None or r < min_rank:
                        min_rank = r
                        min_pair = p
            if min_pair is None:
                break

            left, right = min_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == left and word[i+1] == right:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        return word

    def encode(self, text: str):
        ids = []
        chunks = [text]
        if self._split_pattern is not None:
            chunks = re.split(self._split_pattern, text)
        for chunk in chunks:
            if chunk == "":
                continue
            if self.special_tokens is not None and chunk in self.special_tokens:
                b = chunk.encode("utf-8")
                token_id = self.token_to_id.get(b)
                if token_id is None:
                    raise KeyError(f"Special token {chunk!r} not found in vocab bytes.")
                ids.append(token_id)
                continue
            matches = re.findall(self._pattern, chunk)
            for match in matches:
                token_bytes = match.encode("utf-8")
                if token_bytes in self.token_to_id:
                    ids.append(self.token_to_id[token_bytes])
                    continue
                bpe_bytes = self.bpe_on_token_bytes(token_bytes)
                for bs in bpe_bytes:
                    tid = self.token_to_id.get(bs)
                    if tid is not None:
                        ids.append(tid)
                    else:
                        for b in bs:
                            ids.append(self._byte_cache[b])
        return ids
    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            encoded = self.encode(text)
            for token_id in encoded:
                yield token_id

    def decode(self, ids: list[int]):
        text = b"".join(self.vocab[i] for i in ids)
        return text.decode("utf-8", errors="replace")

# if __name__ == '__main__':
#     from tqdm import tqdm
#     import numpy as np
#
#     tokenizer = Tokenizer.from_files(
#         "vocab_TinyStoriesV2-GPT4-train.pkl",
#         "merges_TinyStoriesV2-GPT4-train.pkl",
#         ["<|endoftext|>"]
#     )
    #
    # buffer = []
    # batch_size = 65536
    #
    # # 先统计总行数，用于 tqdm 显示百分比（可选）
    # with open("../data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    #     total_lines = sum(1 for _ in f)
    #
    # with open("../data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as origin, \
    #         open("../data/TinyStoriesV2-GPT4-train_tokens.bin", "ab") as dest:
    #
    #     # tqdm 按行显示进度
    #     for line in tqdm(origin, total=total_lines, desc="Encoding"):
    #         for token_id in tokenizer.encode_iterable([line]):
    #             buffer.append(token_id)
    #             if len(buffer) >= batch_size:
    #                 np.array(buffer, dtype=np.uint16).tofile(dest)
    #                 buffer.clear()
    #
    #     # 写入剩余 token
    #     if buffer:
    #         np.array(buffer, dtype=np.uint16).tofile(dest)
    # with open("../data/TinyStoriesV2-GPT4-train_tokens.bin", "rb") as f:
    #     while True:
    #         chunk_bytes = f.read(65536 * 2)  # 每个 token 2 字节
    #         if not chunk_bytes:
    #             break
    #         token_ids = np.frombuffer(chunk_bytes, dtype=np.uint16)
    #         text = tokenizer.decode(token_ids.tolist())
    #         print(text)