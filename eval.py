import os
import re
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from train import LemmaTransformer, GlossSimilarity, WindowFeatures, WordNetOverlap

# Load embedder only once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Helper Functions ===
def load_test_sentences(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def load_gold_labels(path):
    with open(path, encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip() in ("1", "2")]

def evaluate_model(model_path, test_path, label_path, debug=False):
    sents = load_test_sentences(test_path)
    gold = load_gold_labels(label_path)
    model = joblib.load(model_path)

    if "deed" in model_path.lower():
        X_test = sents
    else:
        X_test = embedder.encode(sents, convert_to_numpy=True)

    preds = model.predict(X_test)

    print(f"\n=== {os.path.basename(model_path)} on {os.path.basename(test_path)} ===")
    print(f"Test size: {len(sents)} sentences")
    print("Accuracy:", f"{accuracy_score(gold, preds):.2%}")
    print(classification_report(gold, preds, digits=3))

    if debug:
        print("\nSentence-by-sentence predictions:")
        for i, (s, g, p) in enumerate(zip(sents, gold, preds)):
            correctness = "✅" if g == p else "❌"
            print(f"{i+1:02d}. [{correctness}] GOLD: {g} | PRED: {p} | {s}")

# === Main Evaluation Loop ===
def run_all_tests_for_word(word, debug = False):
    model_file_map = {
        "camper": "camper_mlp_sbert_aug.joblib",
        "conviction": "conviction_mlp_sbert.joblib",
        "deed": "deed_stack_pipe.joblib",
    }

    model_path = model_file_map[word.lower()]
    pattern = re.compile(rf"{word}(_\w+)?_test\.txt$")
    all_files = os.listdir(".")

    for test_file in sorted(f for f in all_files if pattern.fullmatch(f)):
        base = re.sub(r"\.txt$", "", test_file)
        label_file = f"result_{base}.txt"

        if not os.path.exists(label_file):
            print(f"\n⚠️  No matching label file found for: {label_file}")
            continue
        if debug:
            evaluate_model(model_path, test_file, label_file, debug=True)
        else:
            evaluate_model(model_path, test_file, label_file)

# Example usage:
if __name__ == "__main__":
    for w in ["camper", "conviction", "deed"]:
        print(f"\n=== {w.upper()} ===")
        run_all_tests_for_word(w, debug = True)
