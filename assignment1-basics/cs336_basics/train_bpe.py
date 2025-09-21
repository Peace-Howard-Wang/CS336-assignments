import pickle
import time
from functools import partial
from multiprocessing import Pool
import os
import regex as re
import collections
from typing import BinaryIO

def logger_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[INFO]: {func.__name__}运行时间: {end - start:.4f} 秒")
        return result
    return wrapper

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_chunk_tasks(train_file_path, split_special_token):
    num_processes = os.cpu_count() - 2 or 1
    with open(train_file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

    chunk_tasks = list(zip(boundaries[:-1], boundaries[1:]))
    print(chunk_tasks)
    return chunk_tasks


def process_chunk(train_file_path, special_tokens, start_end_tuple):
    start_byte, end_byte = start_end_tuple
    pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    with open(train_file_path, "rb") as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)
    chunk_text = chunk_bytes.decode("utf-8")

    split_pattern = "|".join(re.escape(st) for st in special_tokens)
    chunks = re.split(split_pattern, chunk_text)

    vocab = collections.defaultdict(int)

    for sub_chunk in chunks:
        matches = re.finditer(pattern, sub_chunk)
        for match in matches:
            vocab[match.group(0)] += 1

    return vocab

@logger_time
def run_pretokenization(train_file_path, special_tokens):
    split_special_token = special_tokens[0].encode('utf-8')
    chunk_tasks = get_chunk_tasks(train_file_path, split_special_token)
    num_processes = len(chunk_tasks)
    print(f"文件被分成了 {num_processes} 个数据块")

    with Pool(num_processes) as pool:
        func = partial(process_chunk, train_file_path, special_tokens)
        all_results = pool.map(func, chunk_tasks)
    final_words = collections.defaultdict(int)
    for word_counts in all_results:
        for word, count in word_counts.items():
            final_words[tuple([bytes([b]) for b in word.encode('utf-8')])] += count


    print(f"总共找到 {len(final_words)} 个预分词单元")
    return final_words



def get_stats(word_counts):
    """初始化 pair 计数"""
    pairs = collections.defaultdict(int)
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += count
    return pairs

def merge_vocab(pair, word_counts, pair_counts=None):
    """
    pair: (first_byte, second_byte)
    word_counts: {tuple_of_bytes: count}
    pair_counts: 可选，增量更新时提供 pair_counts
    """
    if pair_counts is not None:
        pair_counts.pop(pair, None)
    merge_word_counts = collections.defaultdict(int)
    first, second = pair
    for word, count in word_counts.items():
        i = 0
        new_word = []
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                new_word.append(first + second)
                if pair_counts is not None:
                    if i > 0:
                        left_pair = (word[i-1], word[i])
                        pair_counts[left_pair] -= count
                        pair_counts[(word[i-1], first+second)] += count
                    if i+2 < len(word):
                        right_pair = (word[i+1], word[i+2])
                        pair_counts[right_pair] -= count
                        pair_counts[(first + second, word[i + 2])] += count

                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merge_word_counts[tuple(new_word)] += count
    return merge_word_counts

@logger_time
def train_bpe(input_path, vocab_size, special_tokens, incre=False):
    word_counts = run_pretokenization(input_path, special_tokens)
    vocab = collections.defaultdict(bytes)
    merges = []
    next_id = 0

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    for i in range(256):
        if next_id >= vocab_size:
            break
        vocab[next_id] = bytes([i])
        next_id += 1

    # 初始化 pair_counts
    pair_counts = get_stats(word_counts)

    while len(vocab) < vocab_size and pair_counts:
        best = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair[0], pair[1])
        )
        merges.append(best)
        if incre:
            word_counts = merge_vocab(best, word_counts, pair_counts)
        else:
            word_counts = merge_vocab(best, word_counts)
            pair_counts = get_stats(word_counts)
        new_token = best[0] + best[1]
        vocab[next_id] = new_token
        next_id += 1
        pair_counts = collections.defaultdict(int, {p: c for p, c in pair_counts.items() if c > 0})


    return vocab, merges




if __name__ == '__main__':
    input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = train_bpe(input_path, 1000, ["<|endoftext|>"], True)
    name = input_path.split("/")[-1].replace("txt", "pkl")
    # 保存
    with open("vocab_" + name, "wb") as f:
        pickle.dump(vocab, f)

    with open("merges_" + name,"wb") as f:
        pickle.dump(merges, f)

    # with open("vocab_TinyStoriesV2-GPT4-train.pkl", "rb") as f:
    #     vocab_loaded = pickle.load(f)
    #
    # with open("merges_TinyStoriesV2-GPT4-train.pkl", "rb") as f:
    #     merges_loaded = pickle.load(f)
    # print(vocab_loaded)
    # print(merges_loaded)
