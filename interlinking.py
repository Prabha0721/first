import streamlit as st
import pandas as pd
import numpy as np
import cosine_similarity

# Load embeddings
@st.cache_data
def load_embeddings(file_path):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_json(file_path)
    
    # Convert embedding columns to NumPy arrays
    if "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    else:
        embedding_columns = df.columns[1:]  # Assuming first column is URL
        df["embedding"] = df[embedding_columns].apply(lambda row: np.array(row, dtype=float), axis=1)
    
    return df

# Find similar pages based on cosine similarity
def find_similar_pages(url, df, top_n=5):
    query_embedding = df[df["url"] == url]["embedding"].values

    if len(query_embedding) == 0:
        return None  # URL not found in dataset
    
    query_embedding = query_embedding[0].reshape(1, -1)  # Reshape for similarity calculation
    all_embeddings = np.vstack(df["embedding"].values)  # Convert list of arrays to 2D NumPy array

    similarities = cosine_similarity(query_embedding, all_embeddings)[0]
    df["similarity"] = similarities  # Add similarity scores
    df_sorted = df.sort_values(by="similarity", ascending=False)  # Sort by similarity

    return df_sorted[["url", "similarity"]].head(top_n + 1)[1:]  # Exclude the original URL

# Streamlit UI
st.title("Internal Linking Suggestion Tool")
uploaded_file = st.file_uploader("Upload Sitemap Embeddings File (CSV/JSON)", type=["csv", "json"])

if uploaded_file is not None:
    df = load_embeddings(uploaded_file)

    url_input = st.text_input("Enter URL to Find Related Pages")
    
    if url_input:
        similar_pages = find_similar_pages(url_input, df)
        
        if similar_pages is None:
            st.error("URL not found in dataset.")
        else:
            st.success("Here are the most relevant internal linking suggestions:")
            st.dataframe(similar_pages)

