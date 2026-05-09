# Assignment 1: AG News Text Classification with Embeddings

**Course:** 5LN712 Information Retrieval  
**Author:** Chetan (Sintooop)  
**Date:** May 2026

---

## 1. Problem Definition

News retrieval systems need to route articles to the correct topic category before search or filtering can be applied effectively. Without accurate topic classification, users receive irrelevant results and retrieval quality degrades. This project addresses the challenge of classifying short news texts into one of four categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

The embedding challenge is that all four categories share general vocabulary, but differ in domain-specific terminology, named entities, and context. For example, a business article about a technology company can overlap with Sci/Tech, making surface-level keyword matching insufficient.

---

## 2. Dataset

- **Source:** AG News corpus (fancyzhx/ag_news on Hugging Face)
- **Subset:** 4,000 training samples and 800 test samples (1,000 and 200 per class respectively), stratified by label
- **Classes:** World (0), Sports (1), Business (2), Sci/Tech (3)
- **Hugging Face dataset:** https://huggingface.co/datasets/Sintooop/ag-news-subset

The dataset was subsampled to keep training time reasonable while maintaining class balance.

---

## 3. Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

Each text is tokenized and passed through the model. The CLS token from the last hidden layer is extracted as a 384-dimensional vector representing the semantic content of the text. Embeddings are then standardized using a StandardScaler before being passed to the classifier.

The embedding model is frozen — no fine-tuning was performed. Only the downstream classifier is trained for this task.

---

## 4. Classifiers and Evaluation

Three classifiers were trained and evaluated on the held-out test set (random seed: 712):

| Model               | Accuracy | Macro F1 | Weighted F1 |
|---------------------|----------|----------|-------------|
| Logistic Regression | 0.7600   | 0.7600   | 0.7600      |
| Random Forest       | 0.8100   | 0.8100   | 0.8100      |
| **SVM (best)**      | **0.8588** | **0.8600** | **0.8600** |

SVM with an RBF kernel performed best, achieving 85.9% accuracy. Sports was the easiest category to classify (F1: 0.94), while Business and Sci/Tech showed more confusion with each other.

---

## 5. Links

- **GitHub:** https://github.com/Sintooop/ag-news-classifier
- **Dataset:** https://huggingface.co/datasets/Sintooop/ag-news-subset
- **Model:** https://huggingface.co/Sintooop/ag-news-classifier
- **Demo:** https://huggingface.co/spaces/Sintooop/ag-news-demo

---

## 6. Reflection on Working with AI Tools

Claude was used extensively throughout this assignment to scaffold the training pipeline, debug environment issues on the Uppmax HPC cluster, and generate the demo code. The AI was particularly helpful for resolving SLURM job submission errors, fixing package conflicts, and structuring the overall project.

However, all code was reviewed and tested manually. Understanding what each component does was essential for debugging and explaining the results. The AI accelerated development significantly but required careful verification at each step, especially when adapting code to the specific server environment.

---

## References

Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *Advances in Neural Information Processing Systems*, 28.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *Proceedings of EMNLP 2019*.
