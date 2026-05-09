import numpy as np
import json
import csv
import joblib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import torch
from collections import defaultdict

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
SEED         = 712

# 1. Load dataset
print("Loading AG News...")
ds = load_dataset("fancyzhx/ag_news")

def subsample(split, n):
    buckets = defaultdict(list)
    for item in ds[split]:
        buckets[item["label"]].append(item)
    out = []
    for items in buckets.values():
        out.extend(items[:n])
    return out

train_items  = subsample("train", 1000)
test_items   = subsample("test",  200)
train_texts  = [x["text"]  for x in train_items]
train_labels = [x["label"] for x in train_items]
test_texts   = [x["text"]  for x in test_items]
test_labels  = [x["label"] for x in test_items]
print(f"Train: {len(train_texts)}  Test: {len(test_texts)}")

# 2. Generate embeddings
print(f"Loading embedding model: {EMBED_MODEL}")
tokenizer   = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)
embed_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = embed_model.to(device)
print(f"Device: {device}")

def get_embeddings(texts, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt",
                        truncation=True, max_length=128, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = embed_model(**enc)
        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(emb)
        if i % 500 == 0:
            print(f"  Processing {i}/{len(texts)}")
    return np.vstack(all_embs)

print("Generating train embeddings...")
X_train = get_embeddings(train_texts)
print("Generating test embeddings...")
X_test  = get_embeddings(test_texts)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 3. Train classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
    "SVM":                SVC(kernel="rbf", C=1.0, random_state=SEED),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
}

all_metrics = {}
best_model, best_acc, best_name = None, 0, ""

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    acc   = accuracy_score(test_labels, preds)
    f1    = f1_score(test_labels, preds, average="macro")
    wf1   = f1_score(test_labels, preds, average="weighted")
    print(classification_report(test_labels, preds, target_names=LABEL_NAMES))
    all_metrics[name] = {
        "accuracy":    round(acc, 4),
        "macro_f1":    round(f1,  4),
        "weighted_f1": round(wf1, 4)
    }
    if acc > best_acc:
        best_acc, best_model, best_name = acc, clf, name

print(f"\nBest model: {best_name}  Accuracy: {best_acc:.4f}")

# 4. Save outputs
joblib.dump({"clf": best_model, "scaler": scaler}, "model.joblib")
print("Model saved to model.joblib")

with open("metrics.json", "w") as f:
    json.dump({"best_model": best_name,
               "accuracy":   round(best_acc, 4),
               "all_models": all_metrics}, f, indent=2)
print("Metrics saved to metrics.json")

preds_best = best_model.predict(X_test)
with open("predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "true_label", "predicted_label"])
    for txt, true, pred in zip(test_texts[:20],
                                test_labels[:20],
                                preds_best[:20]):
        w.writerow([txt[:80], LABEL_NAMES[true], LABEL_NAMES[pred]])
print("Predictions saved to predictions.csv")
print("\nDone!")
