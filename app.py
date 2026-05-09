import gradio as gr
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_REPO = "Sintooop/ag-news-classifier"

print("Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)
embed_model.eval()

print("Loading classifier...")
model_path = hf_hub_download(repo_id=HF_REPO, filename="model.joblib")
bundle = joblib.load(model_path)
clf, scaler = bundle["clf"], bundle["scaler"]

def predict(text):
    if not text.strip():
        return "Please enter some text."
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        out = embed_model(**enc)
    emb = out.last_hidden_state[:, 0, :].numpy()
    emb = scaler.transform(emb)
    pred = clf.predict(emb)[0]
    return LABEL_NAMES[pred]

examples = [
    ["Apple announces new iPhone with AI features"],
    ["Brazil wins the World Cup final against France"],
    ["Federal Reserve raises interest rates again"],
    ["NASA discovers water on Mars surface"],
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter a news headline or short text", lines=3),
    outputs=gr.Label(label="Predicted category"),
    title="AG News Text Classifier",
    description="Enter a short news text and the model will predict its category.",
    examples=examples,
)

demo.launch()
