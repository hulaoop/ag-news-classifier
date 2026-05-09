# AG News Text Classifier

Text classification using sentence embeddings and scikit-learn classifiers on the AG News dataset.

## Links

- Dataset: https://huggingface.co/datasets/Sintooop/ag-news-subset
- Model: https://huggingface.co/Sintooop/ag-news-classifier
- Demo: https://huggingface.co/spaces/Sintooop/ag-news-demo

## Task

Given a short news text, predict its category:

| Label | Category |
|-------|----------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.7600   |
| Random Forest       | 0.8100   |
| SVM (best)          | 0.8588   |

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run the demo locally:

```bash
python app.py
```

## Files

- `train.py` — loads AG News, generates embeddings, trains and evaluates classifiers
- `app.py` — Gradio web demo
- `requirements.txt` — Python dependencies
- `report.md` — assignment report
- `metrics.json` — evaluation results
- `predictions.csv` — sample predictions on test set

## Method

1. Load AG News subset (4,000 train / 800 test)
2. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
3. Standardize embeddings with StandardScaler
4. Train Logistic Regression, SVM, and Random Forest classifiers
5. Evaluate on held-out test set
