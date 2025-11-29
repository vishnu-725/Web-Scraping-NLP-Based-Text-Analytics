import os
import re
import math
import string
from typing import List, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup



INPUT_XLSX = r"C:\Users\vishn\Downloads\drive-download-20251127T142747Z-1-001\Input.xlsx"
OUTPUT_XLSX = r"C:\Users\vishn\Downloads\drive-download-20251127T142747Z-1-001\final_output_blackcoffer.xlsx"

ARTICLES_DIR = r"C:\Users\vishn\Downloads\drive-download-20251127T142747Z-1-001\articles"

STOPWORDS_PATTERN = r"C:\Users\vishn\Downloads\drive-download-20251127T142747Z-1-001\StopWords\StopWords_*.txt"

POSITIVE_WORDS_FILE = r"C:\Users\vishn\Downloads\drive-download-20251127T142747Z-1-001\MasterDictionary\positive-words.txt"
NEGATIVE_WORDS_FILE = r"C:\Users\vishn\Downloads\drive-download-20251127T142747Z-1-001\MasterDictionary\negative-words.txt"



def load_stopwords(pattern: str = STOPWORDS_PATTERN) -> set:

    import glob

    stopwords = set()
    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="ISO-8859-1") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token = line.split("|")[0].strip().lower()
                if token:
                    stopwords.add(token)
    return stopwords


def load_master_dictionary(pos_file: str, neg_file: str) -> Tuple[set, set]:
    def load_words(path: str) -> set:
        words = set()
        with open(path, "r", encoding="ISO-8859-1") as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith(";"):
                    words.add(w)
        return words

    pos = load_words(pos_file)
    neg = load_words(neg_file)
    return pos, neg


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}


def extract_article_from_url(url: str) -> Tuple[str, str]:

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch URL: {url} | {e}")
        return "", ""

    soup = BeautifulSoup(resp.content, "html.parser")

    title_tag = soup.find("h1")
    if title_tag and title_tag.get_text(strip=True):
        title = title_tag.get_text(separator=" ", strip=True)
    elif soup.title:
        title = soup.title.get_text(separator=" ", strip=True)
    else:
        title = ""

    article_text = ""
    article_tag = soup.find("article")
    if article_tag:
        paragraphs = article_tag.find_all("p")
        article_text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
    else:
        paragraphs = soup.find_all("p")
        article_text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

    article_text = re.sub(r"\s+", " ", article_text).strip()

    return title, article_text


def save_article_text(url_id: str, title: str, text: str, out_dir: str = ARTICLES_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"{url_id}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        if title:
            f.write(title + "\n\n")
        f.write(text)
    return file_path



def tokenize_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zA-Z]+", text)


def count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    if not word:
        return 0

    syllables = 0
    prev_char_vowel = False
    for ch in word:
        if ch in vowels:
            if not prev_char_vowel:
                syllables += 1
            prev_char_vowel = True
        else:
            prev_char_vowel = False

    if word.endswith(("es", "ed")) and syllables > 1:
        syllables -= 1

    return syllables if syllables > 0 else 1


def count_personal_pronouns(text: str) -> int:
    text_low = text.lower()
    pronouns = re.findall(r"\b(i|we|my|ours|us)\b", text_low)
    return len(pronouns)




def analyze_text(text: str, stopwords: set, pos_dict: set, neg_dict: set) -> dict:
    sentences = tokenize_sentences(text)
    num_sentences = len(sentences) if sentences else 1

    raw_tokens = tokenize_words(text)
    cleaned_tokens = [w for w in raw_tokens if w not in stopwords]
    num_words = len(cleaned_tokens) if cleaned_tokens else 1

    positive_score = 0
    negative_score = 0

    for w in cleaned_tokens:
        if w in pos_dict:
            positive_score += 1
        elif w in neg_dict:
            negative_score += 1

    negative_score = abs(negative_score)

    polarity_score = (positive_score - negative_score) / (
        (positive_score + negative_score) + 0.000001
    )

    subjectivity_score = (positive_score + negative_score) / (num_words + 0.000001)

    avg_sentence_length = num_words / num_sentences

    complex_words = [w for w in cleaned_tokens if count_syllables(w) > 2]
    complex_word_count = len(complex_words)
    percentage_complex_words = complex_word_count / num_words

    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    avg_words_per_sentence = avg_sentence_length

    word_count = num_words

    total_syllables = sum(count_syllables(w) for w in cleaned_tokens)
    syllables_per_word = total_syllables / num_words

    personal_pronouns = count_personal_pronouns(text)

    total_chars = sum(len(w) for w in cleaned_tokens)
    avg_word_length = total_chars / num_words

    return {
        "POSITIVE SCORE": positive_score,
        "NEGATIVE SCORE": negative_score,
        "POLARITY SCORE": polarity_score,
        "SUBJECTIVITY SCORE": subjectivity_score,
        "AVG SENTENCE LENGTH": avg_sentence_length,
        "PERCENTAGE OF COMPLEX WORDS": percentage_complex_words,
        "FOG INDEX": fog_index,
        "AVG NUMBER OF WORDS PER SENTENCE": avg_words_per_sentence,
        "COMPLEX WORD COUNT": complex_word_count,
        "WORD COUNT": word_count,
        "SYLLABLE PER WORD": syllables_per_word,
        "PERSONAL PRONOUNS": personal_pronouns,
        "AVG WORD LENGTH": avg_word_length,
    }




def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"{INPUT_XLSX} not found.")

    df_input = pd.read_excel(INPUT_XLSX)

    if "URL_ID" not in df_input.columns or "URL" not in df_input.columns:
        raise ValueError("Input.xlsx must contain 'URL_ID' and 'URL' columns.")

    print("[INFO] Loading stopwords and dictionaries...")
    stopwords = load_stopwords()
    pos_dict, neg_dict = load_master_dictionary(POSITIVE_WORDS_FILE, NEGATIVE_WORDS_FILE)

    print("[INFO] Extracting articles from URLs...")
    texts_by_id = {}

    for idx, row in df_input.iterrows():
        url_id = str(row["URL_ID"]).strip()
        url = str(row["URL"]).strip()

        print(f"[INFO] ({idx+1}/{len(df_input)}) Processing URL_ID={url_id}")

        title, article_text = extract_article_from_url(url)

        if not article_text:
            print(f"[WARN] No article for URL_ID={url_id}")
            texts_by_id[url_id] = ""
            continue

        save_article_text(url_id, title, article_text, ARTICLES_DIR)

        full_text = (title + "\n\n" + article_text).strip()
        texts_by_id[url_id] = full_text

    print("[INFO] Analyzing text...")
    result_rows = []

    for idx, row in df_input.iterrows():
        url_id = str(row["URL_ID"]).strip()
        text = texts_by_id.get(url_id, "")

        if not text:
            metrics = {
                "POSITIVE SCORE": 0,
                "NEGATIVE SCORE": 0,
                "POLARITY SCORE": 0.0,
                "SUBJECTIVITY SCORE": 0.0,
                "AVG SENTENCE LENGTH": 0.0,
                "PERCENTAGE OF COMPLEX WORDS": 0.0,
                "FOG INDEX": 0.0,
                "AVG NUMBER OF WORDS PER SENTENCE": 0.0,
                "COMPLEX WORD COUNT": 0,
                "WORD COUNT": 0,
                "SYLLABLE PER WORD": 0.0,
                "PERSONAL PRONOUNS": 0,
                "AVG WORD LENGTH": 0.0,
            }
        else:
            metrics = analyze_text(text, stopwords, pos_dict, neg_dict)

        row_dict = row.to_dict()
        row_dict.update(metrics)
        result_rows.append(row_dict)

    df_output = pd.DataFrame(result_rows)

    desired_cols = [
        "URL_ID",
        "URL",
        "POSITIVE SCORE",
        "NEGATIVE SCORE",
        "POLARITY SCORE",
        "SUBJECTIVITY SCORE",
        "AVG SENTENCE LENGTH",
        "PERCENTAGE OF COMPLEX WORDS",
        "FOG INDEX",
        "AVG NUMBER OF WORDS PER SENTENCE",
        "COMPLEX WORD COUNT",
        "WORD COUNT",
        "SYLLABLE PER WORD",
        "PERSONAL PRONOUNS",
        "AVG WORD LENGTH",
    ]

    existing_cols = [c for c in df_output.columns if c not in desired_cols]

    final_cols = []
    for c in df_input.columns:
        if c not in final_cols:
            final_cols.append(c)

    for c in desired_cols:
        if c not in final_cols:
            final_cols.append(c)

    for c in existing_cols:
        if c not in final_cols:
            final_cols.append(c)

    df_output = df_output[final_cols]

    print(f"[INFO] Writing output to {OUTPUT_XLSX} ...")
    df_output.to_excel(OUTPUT_XLSX, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
