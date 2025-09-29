from random import random

from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext

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

if __name__ == "__main__":
    from fastwarc.warc import ArchiveIterator, WarcRecordType
    from tqdm import tqdm  # 进度条
    import random

    english_count = 0
    total_count = 0
    sample_for_manual_check = []

    with open("../data/example.warc.gz", "rb") as f:
        for record in tqdm(ArchiveIterator(f), desc="Processing WARC"):
            if record.record_type != WarcRecordType.response:
                continue
            raw_bytes = record.reader.read()
            text = extract_text_from_html_bytes(raw_bytes)
            if not text.strip():
                continue
            lang_code, confidence = identify_language(text)
            total_count += 1
            if lang_code == "en":
                english_count += 1

            if len(sample_for_manual_check) < 20 and random.random() < 0.1:
                sample_for_manual_check.append((text[:200], lang_code, confidence))
                print(len(sample_for_manual_check))
            if len(sample_for_manual_check) >= 20:
                break
    english_fraction = english_count / total_count if total_count > 0 else 0
    print(f"English fraction: {english_fraction:.2%} ({english_count}/{total_count})")
    print("Sample for manual check:")
    for text_snippet, lang_code, confidence in sample_for_manual_check:
        print(f"[{lang_code} ({confidence:.2%})] {text_snippet}")