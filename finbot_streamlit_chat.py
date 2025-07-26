
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Load model and data
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
kb_df = pd.read_csv("knowledge_base_top_50_improved.csv")
X_narrative = np.load("final_complaint_embeddings.npy")
kmeans = KMeans(n_clusters=50, random_state=42)
kmeans.fit(X_narrative)

def embed_user_input(text):
    return embed_model.encode([text])

def find_best_kb_response(user_input):
    embedding = embed_user_input(user_input)
    cluster_id = kmeans.predict(embedding)[0] + 1
    match = kb_df[kb_df['Cluster'] == cluster_id]
    if not match.empty:
        row = match.iloc[0]
        response = row['Suggested Response']
        action = row['Suggested Action']
        return f"\n**Response:** {response}\n\n**Next Step:** {action}"
    else:
        return "⚠️ Sorry, we couldn't find a matching resolution. Please try rephrasing."

st.set_page_config(page_title="FinBot - Complaint Assistant", layout="centered")
st.title("🤖 FinBot - Smart Complaint Resolver")
st.markdown("Type a financial complaint below. FinBot will analyze it and give a resolution suggestion.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your complaint..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    response = find_best_kb_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
