import os
import joblib
from sentence_transformers import SentenceTransformer
from train import LemmaTransformer, GlossSimilarity, WindowFeatures, WordNetOverlap # Required for Models

def WSD_Test_camper(sentences):
    """
    Takes a list of sentences containing 'camper' and returns predicted sense labels (1 or 2).
    Loads the SBERT embedder and camper model inside the function as required.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    model = joblib.load("camper_mlp_sbert_aug.joblib")
    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    predictions = model.predict(embeddings)
    return predictions.tolist()

def WSD_Test_conviction(sentences):
    """
    Takes a list of sentences containing 'conviction' and returns predicted sense labels (1 or 2).
    Loads the SBERT embedder and conviction model inside the function.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    model = joblib.load("conviction_mlp_sbert.joblib")
    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    predictions = model.predict(embeddings)
    return predictions.tolist()

def WSD_Test_deed(sentences):
    """
    Takes a list of sentences containing 'deed' and returns predicted sense labels (1 or 2).
    Loads the pipeline model (which includes its own preprocessing).
    """
    model = joblib.load("deed_stack_pipe.joblib")
    predictions = model.predict(sentences)
    return predictions.tolist()

def run_test(firstname, lastname):
    name_tag = f"{firstname.lower()}{lastname.lower()}"
    test_files = {
        "camper": "camper_test.txt",
        "conviction": "conviction_test.txt",
        "deed": "deed_test.txt",
    }

    functions = {
        "camper": WSD_Test_camper,
        "conviction": WSD_Test_conviction,
        "deed": WSD_Test_deed,
    }

    for word, file in test_files.items():
        if not os.path.exists(file):
            print(f"Missing test file: {file}")
            continue

        # Load sentences
        with open(file, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        # Run prediction
        print(f"Predicting senses for {word}...")
        results = functions[word](sentences)

        # Save to file
        out_file = f"result_{word}_{name_tag}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for label in results:
                f.write(str(label) + "\n")

        print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    run_test("Harley", "Gribble")