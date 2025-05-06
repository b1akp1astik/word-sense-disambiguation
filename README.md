# Word Sense Disambiguation – CS 5322 Program 3

This project implements a supervised **Word Sense Disambiguation (WSD)** system for three target words: `camper`, `conviction`, and `deed`. Each word has two defined senses, and the system classifies which sense applies to a given sentence.

## Overview

- Each word is trained using a separate model.
- Models are built using either **Sentence-BERT embeddings** or symbolic pipelines (e.g., WordNet gloss features).
- Final classification outputs are integers `1` or `2` per sentence, representing the predicted sense.

## Files

- `cs5322s25.py`  
  Main module. Contains `WSD_Test_camper`, `WSD_Test_conviction`, and `WSD_Test_deed` functions. Loads models and returns predictions.

- `train.py`  
  Contains data loading, feature engineering classes (e.g., `GlossSimilarity`, `LemmaTransformer`), and model training scripts for each word. Run directly to retrain models.

- `eval.py`  
  Contains evaluation utilities and helpers for testing predictions against ground truth.

- `test.py`  
  Local script used to test the `WSD_Test_*` functions manually.

- `*.joblib`  
  Pre-trained model files used by `cs5322s25.py`. Required for predictions.

## How to Use

1. Install dependencies (Python 3.8+):

```bash
pip install -r requirements.txt
```
2. To run predictions:

```python
from cs5322s25 import WSD_Test_camper
sentences = ["He parked his camper near the lake.", "Each camper got a meal pass."]
print(WSD_Test_camper(sentences))  # Example: [1, 2]
```

3. To regenerate models:

```bash
python train.py
```

4. To run the final batch test output:

```bash
python cs5322s25.py
```

## Requirements

- scikit-learn

- nltk

- sentence-transformers

- joblib

## Notes

- All models are loaded inside the prediction functions, as required by the assignment.

- Additional features like synonym augmentation and symbolic gloss comparison were used to improve accuracy on difficult cases (especially for deed).

*Developed by Harley Gribble for CS 5322 – Spring 2025.*