import joblib
import numpy as np
import nltk
import re
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sentence_transformers import SentenceTransformer

# Download required NLTK data
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("wordnet")
nltk.download("omw-1.4")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
lemmatizer = WordNetLemmatizer()

# ─── Preprocessing & Feature Extractors ─────────────────────────────
class LemmaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [" ".join(lemmatizer.lemmatize(w) for w in word_tokenize(s.lower())) for s in X]

class WordNetOverlap(BaseEstimator, TransformerMixin):
    def __init__(self, word): self.word = word
    def fit(self, X, y=None): return self
    def transform(self, X):
        senses = wn.synsets(self.word, pos=wn.NOUN)[:2]
        gloss_sets = [set(word_tokenize(s.definition().lower())) for s in senses]
        feats = []
        for sent in X:
            toks = set(word_tokenize(sent.lower()))
            feats.append([len(toks & gloss_sets[0]), len(toks & gloss_sets[1])])
        return np.array(feats)

class WindowFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, word): self.word = word
    def fit(self, X, y=None): return self
    def transform(self, X):
        feats = []
        for sent in X:
            toks = word_tokenize(sent.lower())
            idx = toks.index(self.word) if self.word in toks else -1
            window = toks[max(0, idx-2): idx+3]
            feats.append([len(window), sum(1 for t in window if t in ("the", "a", "an"))])
        return np.array(feats)

class GlossSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, word):
        self.word = word

    def fit(self, X, y=None): return self

    def transform(self, X):
        embedder = SentenceTransformer("all-MiniLM-L6-v2")  # reload here, avoid storing
        senses = wn.synsets(self.word, pos=wn.NOUN)[:2]
        gloss_vecs = embedder.encode([s.definition() for s in senses])
        sent_vecs = embedder.encode(X)
        return np.array([[np.dot(s, gloss_vecs[0]), np.dot(s, gloss_vecs[1])] for s in sent_vecs])


# ─── Data Loading & Augmentation ────────────────────────────────────
def load_data(path):
    sents, labs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0] in ("1", "2"):
                labs.append(int(parts[0]))
                sents.append(parts[1])
    return sents, labs

def synonym_augment(sents, labels, n_aug=2):
    aug_sents, aug_labels = [], []
    for s, l in zip(sents, labels):
        tokens = word_tokenize(s)
        for i, word in enumerate(tokens):
            syns = wn.synsets(word)
            if syns:
                lemmas = set(lemma.replace("_", " ") for syn in syns for lemma in syn.lemma_names())
                for lemma in list(lemmas)[:n_aug]:
                    new = tokens[:i] + [lemma] + tokens[i+1:]
                    aug_sents.append(" ".join(new))
                    aug_labels.append(l)
    return sents + aug_sents, labels + aug_labels

# ─── Training Scripts ───────────────────────────────────────────────
def build_conviction_model():
    sents, labels = load_data("conviction_extended.txt")
    X = embedder.encode(sents)
    model = MLPClassifier(hidden_layer_sizes=(50,), alpha=0.0001, max_iter=1000, random_state=42)
    model.fit(X, labels)
    joblib.dump(model, "conviction_mlp_sbert.joblib")
    print("Conviction model saved.")

def build_camper_model():
    sents, labels = load_data("camper_extended.txt")
    s_aug, l_aug = synonym_augment(sents, labels, n_aug=2)
    X = embedder.encode(s_aug)
    model = MLPClassifier(hidden_layer_sizes=(50, 50), alpha=0.0001, max_iter=1000, random_state=42)
    model.fit(X, l_aug)
    joblib.dump(model, "camper_mlp_sbert_aug.joblib")
    print("Camper model saved.")

def build_deed_model():
    sents, labels = load_data("deed_extended.txt")
    s_aug, l_aug = synonym_augment(sents, labels, n_aug=3)
    union = FeatureUnion([
        ("tf_w", TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=1)),
        ("tf_c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)),
        ("wnov", WordNetOverlap("deed")),
        ("wind", WindowFeatures("deed")),
        ("glos", GlossSimilarity("deed")),
    ])
    pipe = Pipeline([
        ("lemma", LemmaTransformer()),
        ("feat", union),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    pipe.fit(s_aug, l_aug)
    joblib.dump(pipe, "deed_stack_pipe.joblib")
    print("Deed model saved.")

if __name__ == "__main__":
    build_conviction_model()
    build_camper_model()
    build_deed_model()
