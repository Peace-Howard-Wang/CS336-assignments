import hashlib
import os
import random
import re
import unicodedata
from collections import defaultdict

import fasttext
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes):
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        if encoding is None:
            encoding = "latin-1"
        html_str = html_bytes.decode(encoding, errors="replace")
    text = extract_plain_text(html_str)
    return text

def identify_language(text):
    path = "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/classifiers/lid.176.bin"
    model = fasttext.load_model(path)
    text = text.replace("\n", " ")
    labels, scores = model.predict(text, k=1)
    lang_code = labels[0].replace("__label__", "")
    lang_map = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "zh": "zh",
        "en": "en",
    }
    lang_code = lang_map.get(lang_code, lang_code)
    confidence = float(scores[0])
    return lang_code, confidence

def mask_emails(text):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    text, count = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return text, count

def mask_phone_numbers(text):
    pattern = r"(?:\+\d{1,3}[-.\s]?)?(?:\(\d{1,4}\)|\d{1,4})(?:[-.\s]?\d{1,4}){1,3}(?:\s*(?:#|x\.?|ext\.?|extension)\s*\d+)?"
    text, count = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return text, count

def mask_ips(text):
    pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    text, count = re.subn(pattern, "|||IP_ADDRESS|||", text)
    return tuple((text, count))

def classify_nsfw(text):
    model_path = "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/classifiers/nsfw.bin"
    model = fasttext.load_model(model_path)
    text = text.replace("\n", " ")
    labels, scores = model.predict(text, k=1)
    label = labels[0].replace("__label__", "")
    score = float(scores[0])
    return label, score

def classify_toxic_speech(text):
    model_path = "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/classifiers/hatespeech.bin"
    model = fasttext.load_model(model_path)
    text = text.replace("\n", " ")
    labels, scores = model.predict(text, k=1)
    label = labels[0].replace("__label__", "")
    score = float(scores[0])
    return label, score

def gopher_quality_filter(text):
    words = text.split()
    words_len = len(words)
    if words_len < 50 or words_len > 100000:
        return False
    char_sum = 0
    for word in words:
        char_sum += len(word)
    avg_word_len = char_sum / words_len
    if avg_word_len < 3 or avg_word_len > 10:
        return False
    one_word_count = 0
    for word in words:
        if len(word) == 1:
            one_word_count += 1
    lines = text.split("\n")
    ellipsis_count = 0
    for line in lines:
        if line.strip().endswith("..."):
            ellipsis_count += 1
    if ellipsis_count / len(lines) > 0.3:
        return False
    if one_word_count / words_len > 0.2:
        return False
    return True

def classify_quality(text):
    """
    给定文本，返回 (label, confidence_score)
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return "cc", 1.0
    model = fasttext.load_model("/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/classifiers/quality_classifier.bin")
    labels, scores = model.predict(text, k=1)
    label = labels[0].replace("__label__", "")
    label = "cc" if label == "low" else "wiki"
    score = float(scores[0])
    print(text, label)
    return label, score


def exact_line_deduplication(input_paths, output_dir):
    # 1. 统计每一行的出现次数
    line_counts = defaultdict(int)
    line_storage = {}  # 存储 hash -> 原始行，用于写回

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.rstrip("\n")  # 去掉换行方便哈希
                h = hashlib.md5(line_stripped.encode("utf-8")).hexdigest()
                line_counts[h] += 1
                if h not in line_storage:  # 保存原始行
                    line_storage[h] = line

    # 2. 重写文件，只保留唯一的行
    os.makedirs(output_dir, exist_ok=True)
    for path in input_paths:
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)

        with open(path, "r", encoding="utf-8") as f_in, \
                open(out_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line_stripped = line.rstrip("\n")
                h = hashlib.md5(line_stripped.encode("utf-8")).hexdigest()
                if line_counts[h] == 1:
                    f_out.write(line)




# ---------------- 工具函数 ---------------- #

def normalize_text(text: str) -> str:
    """文本标准化"""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")  # 去掉音符
    text = re.sub(r"\s+", " ", text)  # 多空格压缩
    text = re.sub(r"[^\w\s]", " ", text)  # 去掉标点符号
    return text.strip()

def get_ngrams(tokens, n):
    """生成词 n-gram 集合"""
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def hash_with_seed(x: str, seed: int) -> int:
    """基于 hashlib.sha1 + seed 的稳定 hash"""
    h = hashlib.sha1((str(seed) + "_" + x).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")  # 64-bit int


# ---------------- MinHash ---------------- #

def compute_minhash_signature(ngrams, num_hashes, seed=0):
    """为一个文档的 n-gram 集合生成 MinHash 签名"""
    sig = []
    for i in range(num_hashes):
        min_val = float("inf")
        for ng in ngrams:
            h = hash_with_seed(" ".join(ng), seed + i)
            if h < min_val:
                min_val = h
        sig.append(min_val)
    return sig


# ---------------- 并查集 ---------------- #

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


# ---------------- 主流程 ---------------- #

def minhash_deduplication(input_paths, num_hashes, num_bands,
                              n, threshold, output_dir, seed=42):
    assert num_hashes % num_bands == 0
    random.seed(seed)

    num_docs = len(input_paths)
    rows_per_band = num_hashes // num_bands

    # Step1: 标准化、分词、生成 n-grams
    docs_tokens = []
    docs_ngrams = []
    for path in input_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = normalize_text(f.read())
            tokens = text.split()
            docs_tokens.append(tokens)
            if len(tokens) < n:
                ngrams = set()
            else:
                ngrams = get_ngrams(tokens, n)
            docs_ngrams.append(ngrams)

    # Step2: MinHash 签名
    signatures = [compute_minhash_signature(ngrams, num_hashes, seed)
                  for ngrams in docs_ngrams]

    # Step3: LSH 桶化
    buckets = defaultdict(list)
    for doc_id, sig in enumerate(signatures):
        for b in range(num_bands):
            start = b * rows_per_band
            end = (b+1) * rows_per_band
            band = tuple(sig[start:end])
            band_key = hash_with_seed(str(band), b)
            buckets[(b, band_key)].append(doc_id)

    # Step4: 候选对生成
    candidates = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) > 1:
            for i in range(len(bucket_docs)):
                for j in range(i+1, len(bucket_docs)):
                    candidates.add((min(bucket_docs[i], bucket_docs[j]),
                                    max(bucket_docs[i], bucket_docs[j])))

    # Step5: 精确 Jaccard + 并查集合并
    uf = UnionFind(num_docs)
    for i, j in candidates:
        sim = jaccard(docs_ngrams[i], docs_ngrams[j])
        if sim >= threshold:
            uf.union(i, j)

    # Step6: 聚类 & 保留
    clusters = defaultdict(list)
    for i in range(num_docs):
        clusters[uf.find(i)].append(i)

    retained = set()
    for comp in clusters.values():
        keep = random.choice(comp)  # 每个簇保留一个
        retained.add(keep)

    # Step7: 写输出
    os.makedirs(output_dir, exist_ok=True)
    for idx, path in enumerate(input_paths):
        if idx in retained:
            filename = os.path.basename(path)
            out_path = os.path.join(output_dir, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as f_in, \
                 open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())

    print(f"保留 {len(retained)}/{num_docs} 个文档，去重完成。输出目录: {output_dir}")
    return retained
