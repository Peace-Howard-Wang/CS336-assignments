import re

import requests
from bs4 import BeautifulSoup
from readability import Document


def download_high_quality():
    input_file = "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/positive_urls.txt"
    output_file = "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/high_quality.txt"
    success_count = 0

    with open(input_file, "r", encoding="utf-8") as f, \
         open(output_file, "a", encoding="utf-8") as out_f:

        for i, line in enumerate(f, 1):
            url = line.strip()
            try:
                response = requests.get(url, timeout=10)
                if "text/html" not in response.headers.get("Content-Type", ""):
                    print(f"[{i}] Skipped non-HTML content: {url}")
                    continue

                response.encoding = response.apparent_encoding
                if response.status_code == 200:
                    try:
                        doc = Document(response.text)
                        html_content = doc.summary()
                    except Exception as e:
                        print(f"[{i}] readability error: {e} - {url}")
                        continue

                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text(separator="\n")
                    text = ''.join(c for c in text if c.isprintable())
                    text = re.sub(r'\n\s*\n', '\n', text).strip()
                    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?，。！？]+', '', text)

                    if text:
                        out_f.write(text + "\n<|endoftext|>\n")
                        success_count += 1
                        print(f"[{i}] Downloaded content from {url}")
                    else:
                        print(f"[{i}] No clean text found: {url}")
                else:
                    print(f"[{i}] Failed: Status {response.status_code} - {url}")
            except requests.RequestException as e:
                print(f"[{i}] Error fetching {url}: {e}")

    print(f"Total successful downloads: {success_count}")

import gzip
import random
import re

def download_low_quality():
    input_file = "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/example.warc.wet.gz"
    output_file = "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/low_quality.txt"
    sample_size = 1000 # 抽取样本数量
    success_count = 0

    records = []
    current_record = []

    # 读取 WET 文件，按段落切分
    with gzip.open(input_file, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if current_record:
                    records.append("\n".join(current_record).strip())
                    current_record = []
            else:
                current_record.append(line.strip())

    print(f"📦 Loaded {len(records)} records from {input_file}")

    # 随机采样
    sampled = random.sample(records, min(sample_size, len(records)))

    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, text in enumerate(sampled, 1):
            # 清理不可打印字符
            text = ''.join(c for c in text if c.isprintable())
            text = re.sub(r'\s+', ' ', text).strip()

            if text and len(text.split()) > 5:  # 至少几个词
                out_f.write(text + "\n<|endoftext|>\n")
                success_count += 1
                print(f"[{i}] Saved low-quality sample")
            else:
                print(f"[{i}] Skipped too short/empty")

    print(f"✅ Total low-quality samples saved: {success_count}")



import fasttext
import os

def train_quality_classifier():
    base_path = "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data"

    high_file = os.path.join(base_path, "high_quality.txt")
    low_file = os.path.join(base_path, "low_quality.txt")
    train_file = os.path.join(base_path, "train_quality.txt")
    model_file = os.path.join(base_path, "quality_classifier.bin")

    # 1. 生成训练文件（合并 + 加标签）
    with open(train_file, "w", encoding="utf-8") as out_f:
        # 高质量
        with open(high_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line != "<|endoftext|>":
                    out_f.write(f"__label__high {line}\n")

        # 低质量
        with open(low_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line != "<|endoftext|>":
                    out_f.write(f"__label__low {line}\n")

    print(f"✅ Training file created: {train_file}")

    # 2. 训练 FastText 分类器
    model = fasttext.train_supervised(
        input=train_file,
        epoch=100,          # 训练轮数，可以调大
        lr=0.05,            # 学习率
        wordNgrams=2,      # 使用 bigram
        dim=100,           # 向量维度
        loss="softmax",    # 损失函数
        verbose=2
    )

    # 3. 保存模型
    model.save_model(model_file)
    print(f"✅ Model saved to {model_file}")

    return model


def run_classify_quality(text, model):
    """
    给定文本，返回 (label, confidence_score)
    """
    labels, scores = model.predict(text, k=1)
    label = labels[0].replace("__label__", "")
    score = float(scores[0])
    return label, score


if __name__ == "__main__":
    # model = train_quality_classifier()
    model = fasttext.load_model(
        "/Users/wanghao/PycharmProjects/CS336/assignment4-data/data/classifiers/quality_classifier.bin")
    # 测试
    print(run_classify_quality("This article discusses the Malaysian election results", model))
    print(run_classify_quality("hiefdhiewuhfiewhfiweufhwieufewrf34qrq4tE34fewfewf", model))