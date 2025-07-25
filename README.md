# FinBot: Intelligent Complaint Classification and Semantic Resolution Assistant

## 📃 Executive Summary

FinBot is an AI-driven financial complaint classification and resolution system designed to automate the triaging, categorization, and response suggestion for consumer financial issues. Powered by sentence embeddings, hierarchical machine learning models, and a semantic knowledge base, FinBot efficiently understands and routes customer complaints using advanced Natural Language Processing (NLP) techniques.

Built using real-world data from the Consumer Financial Protection Bureau (CFPB), FinBot supports:

* AI-powered customer support bots
* Complaint routing systems
* Semantic search in compliance platforms

---

## 🌟 Project Goals

* Classify complaints into Product → Sub-product → Issue
* Construct a semantic knowledge base using KMeans clustering
* Automate response recommendation
* Enable chatbot-ready architecture

---

## 📚 Dataset Overview

**Source:** CFPB Public Complaint Dataset
**Cleaned Size:** 120,000 rows

| Field               | Description                                  |
| ------------------- | -------------------------------------------- |
| Complaint Narrative | User-submitted complaint (free text)         |
| Product             | Broad product category (e.g., "Credit card") |
| Sub-product         | Narrower product category                    |
| Issue               | Specific complaint issue                     |

---

## 🔧 Key Modules

### 1. Sentence Embedding

Use BERT-based sentence transformers to convert text into semantic vectors:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(["I was charged twice for a transaction"])  # returns 384-dim vector
```

### 2. Batch Embedding Pipeline

Embeds fields (`Product`, `Sub-product`, `Issue`) in batches and stores them as `.npy` files.

```python
combined = df['Product'] + ' | ' + df['Sub-product'] + ' | ' + df['Issue']
combined_embeddings = model.encode(combined.tolist(), batch_size=32)
```

### 3. Semantic Clustering for KB

```python
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

kmeans = KMeans(n_clusters=50)
labels = kmeans.fit_predict(X_narrative)
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_narrative)
kb_df = complaints_df.iloc[closest_indices]
```

### 4. Hierarchical Classification

**Level 1:** Predict Product
**Level 2:** Predict Sub-product (conditioned on Product)
**Level 3:** Predict Issue (conditioned on Sub-product)

Each layer is modeled using `LogisticRegression` with its own `LabelEncoder` and training subset.

### 5. KB Response Retrieval

```python
def find_best_kb_response(user_input, kmeans, kb_df):
    embedding = embed_user_input(user_input)
    cluster_id = kmeans.predict(embedding)[0] + 1
    match = kb_df[kb_df['Cluster'] == cluster_id].iloc[0]
    return f"\U0001f9fe Response: {match['Suggested Response']}\n\u2705 Action: {match['Suggested Action']}"
```

---

## 🔬 Evaluation Summary

| Classifier  | Method              | Accuracy |
| ----------- | ------------------- | -------- |
| Product     | Logistic Regression | 85–90%   |
| Sub-product | Per-product models  | 80–88%   |
| Issue       | Per-sub-product     | 75–85%   |

---

## 🌐 Outputs

* Trained classifiers (Product, Sub-product, Issue)
* 50 semantic clusters as knowledge base entries
* Embedding repository for all fields
* Response suggestion engine

---

## 📊 Business Value

* ⏱ Reduces manual triaging from hours to seconds
* 📊 Brings structure and automation to customer support workflows
* ✨ Enables integration with voice/chatbot systems
* 🌟 Supports large-scale regulatory or compliance analytics

---

## ✨ Future Work

* 🔗 Integrate with FastAPI or Gradio for chatbot UI
* 🧠 Use GPT-based RAG (Retrieval-Augmented Generation) for dynamic answers
* 📊 Add dashboards for complaint trends and real-time alerting

---

## 🚀 Summary

FinBot provides a modular, production-ready pipeline for transforming unstructured complaint narratives into structured, actionable insights. With a blend of sentence embeddings, hierarchical classification, and semantic clustering, it enables scalable and intelligent complaint management for financial institutions.
