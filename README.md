# Evaluating Note-Taking Quality with Rule-Based, BERT, and LLM Models

**COM S 5790 Final Project, Iowa State University**
Benjamin Jia Zhiang Kam · Mason Inman · Jesus Soto Gonzalez

This project checks whether a student's lecture notes capture specific instructor-defined **IdeaUnits**. It frames the task as binary classification and compares three NLP approaches that rest on different design ideas:

| Approach | Core idea | Folder |
|---|---|---|
| **Rule-based** | Lexical similarity using token-sorted fuzzy ratio (Levenshtein) | [`rule/`](rule/) |
| BERT | Sequence-pair classifier with temporal alignment | [`bert/`](bert/) |
| LLM | Prompted LLaMA 3 8B Instruct (zero-shot and one-shot) | [`LLM/`](LLM/) |

The full write-up is in [`report/final-report.pdf`](report/final-report.pdf).

> This repository is a fork of [MasonInman29/Evaluation-of-note-taking-NLP](https://github.com/MasonInman29/Evaluation-of-note-taking-NLP). My contribution was the **rule-based model**, which is documented in detail below.

---

## Task

For a lecture topic *t*, we have:
- a set of instructor-defined IdeaUnits *U<sub>t</sub>*, and
- student note segments *S<sub>t</sub>*.

The goal is to learn a function *f(u, s) → y ∈ {0, 1}*. The label *y* says whether segment *s* expresses IdeaUnit *u*.

Supervision is very limited, with only one labeled example per topic. The notes themselves are short and noisy.

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Rule-Based** | **0.81** | **0.93** | 0.63 | **0.75** |
| BERT-Based | 0.72 | 0.72 | **0.66** | 0.69 |
| LLM-Based (one-shot) | 0.65 | 0.51 | 0.58 | 0.54 |

**Main finding:** this dataset strongly rewards surface-form overlap. As a result, the simple lexical baseline achieves the best overall F1. BERT recovers more paraphrased ideas and therefore has the highest recall. Prompt-only LLM inference is brittle on short, incomplete notes.

---

## Rule-Based Model (my contribution)

### How it works

1. **Preprocessing** (`pre_processing.ipynb`): the text is lowercased, lemmatized with spaCy (`en_core_web_sm`), and stripped of punctuation and whitespace tokens.
2. **Token sorting:** the student text and the reference text are tokenized and sorted alphabetically. This removes word-order effects but keeps all lexical content.
3. **Token-sorted fuzzy ratio:** normalized Levenshtein similarity is computed on the sorted strings. Levenshtein distance works at the character level, but sorting the tokens first shifts the focus to lexical overlap rather than syntax.

   $$\text{FuzzyRatio}(R, C) = 1 - \frac{LD(R_s, C_s)}{\max(|R_s|, |C_s|)}$$

   Dividing by the maximum string length keeps scores comparable across answers of different lengths.

4. **Thresholding:** empirically selected thresholds $\tau_2 < \tau_1$ assign each response to one of three tiers:

   | Score | Output | Meaning |
   |---|---|---|
   | ≥ τ₁ | `1` | Correct |
   | τ₂ – τ₁ | `0` | Partially correct |
   | < τ₂ | `-1` | Incorrect |

   For evaluation, the output is **collapsed to binary**: only `1` (Correct) counts as positive. Thresholds are tuned with a grid search in `threshold_testing.ipynb`, and the final setting is τ₁ = 0.60 and τ₂ = 0.59.

The approach uses no semantic inference at all. It relies purely on lexical similarity, which suits short, objective answers.

> **Naming note:** the rule notebooks use the column names `Response` (student text) and `CorrectAnswer` (reference text). These correspond to the note segment and the IdeaUnit in the report.

### Development history

Earlier versions combined fuzzy ratio with Jaccard similarity, bag-of-words cosine similarity, spaCy semantic similarity, and a numeric-match feature. Over successive experiments these extra features were removed. The final model uses **fuzzy ratio alone**. Compared with the earlier combined versions, this raised recall from about 0.56 to 0.63 while precision fell only slightly, from 0.98 to 0.93. The unused helper functions are still present in the notebooks.

### Analysis

- **Strength: precision of 0.93, the highest of all models.** When the model predicts "correct", the prediction is very likely right. This happens because the answers are short and objective, so lexical overlap is a strong signal of correctness.
- **Trade-off: lower recall than BERT.** The model misses correct answers that mean the same thing but use different words.
- **Overall behaviour: conservative.** The model avoids false positives at the cost of recall.

### Assumptions and limitations

- Responses are short, so each token carries a lot of weight.
- Correct responses are assumed to stay lexically close to the reference.
- A reference answer must be available, so reference-free grading is not possible.
- The model does not scale well to longer or more subjective answers.

---

## BERT Model

`bert/bert.py` fine-tunes `bert-base-uncased` as an (IdeaUnit, note) pair classifier.

It uses **temporal alignment**: for each IdeaUnit, it extracts the matching span of the notes, assuming the notes follow lecture order. This reduces truncation bias under the 256-token limit.

Training setup: a seeded 80/20 split of `train.csv`, 4 epochs, learning rate 2e-5, batch size 8, and alignment margin 0.3. The decision threshold is tuned on validation data to maximize F1. On this split, the model reaches ROC-AUC 0.79 and PR-AUC 0.74.

## LLM Model 

`LLM/` prompts **LLaMA 3 8B Instruct** in zero-shot and one-shot settings. Each prediction comes from the next-token logits for `YES` and `NO`. For one-shot prompting, the in-context example for each topic is chosen by lexical overlap. [`LLM/README.md`](LLM/README.md) has full details.

---

## Data

The dataset (`Notes.csv`, `train.csv`, `test.csv`) was **provided through the course and is not redistributed** in this repository. All `data/` directories and `*.csv` files are gitignored.

The rule-based notebooks expect `train.csv` and `test.csv` inside `rule/`. They write their cleaned output to `train_cleaned.csv` and `test_cleaned.csv`.

---

## Setup and Reproduction

Each model has its own dependencies. Use a separate virtual environment for each one.

### Rule-based

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r rule/requirements.txt   # pandas, scikit-learn, spacy, rapidfuzz
python -m spacy download en_core_web_sm
```

Then run the notebooks in `rule/` in this order:

1. `pre_processing.ipynb` lemmatizes and cleans `train.csv` or `test.csv`. Edit the input and output filenames as needed.
2. `threshold_testing.ipynb` grid-searches τ₁ and τ₂ on `train_cleaned.csv`.
3. `main.ipynb` classifies `test_cleaned.csv` and prints accuracy, precision, recall, and F1.

### BERT

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn
cd bert
python bert.py --data_dir <path-to-data> \
  --epochs 4 --align_margin 0.3 --optimize_threshold \
  --output_dir results_temporal_8020_4_align3
python evaluate_results.py   # reads results_temporal_8020_4_align3/seed_42
```

Saved outputs from the reported run are in `bert/results_temporal_8020_4_align3/seed_42/`. They include the metrics, curves, the confusion matrix, and an HTML report.

### LLM

Follow the steps in [`LLM/README.md`](LLM/README.md). You will need a Hugging Face token with access to LLaMA 3.

---

## Repository Structure

```
.
├── rule/      # Rule-based model: preprocessing, threshold search, evaluation notebooks
├── bert/      # BERT classifier with temporal alignment + evaluation script and outputs
├── LLM/       # LLaMA 3 prompting pipeline (own README)
├── report/    # ACL-format LaTeX source and final PDF
└── LICENSE
```

---

## Future Work

- **Hybrid lexical-semantic models** that combine the high precision of the rule-based approach with the higher recall of BERT.
- **Cross-dataset generalization:** testing these models on other datasets, including longer or more subjective answers.

## License

MIT. See [LICENSE](LICENSE).
