pip install --upgrade pip
import streamlit as st
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_sitemap_urls(sitemap_url):
    """Fetch and parse URLs from sitemap.xml"""
    response = requests.get(sitemap_url)
    root = ET.fromstring(response.content)
    urls = [elem.text for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    return urls

def get_text_from_url(url):
    """Scrape and extract visible text from a webpage"""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text.strip()
    except Exception as e:
        return ""

def generate_embeddings(urls):
    """Generate embeddings for all URLs"""
    embeddings = {}
    for url in urls:
        text = get_text_from_url(url)
        if text:
            embeddings[url] = model.encode(text)
    return embeddings

def find_related_pages(target_url, url_list, embedding_matrix, top_n=3):
    """Find top-N related pages based on cosine similarity"""
    target_idx = url_list.index(target_url)
    similarities = cosine_similarity([embedding_matrix[target_idx]], embedding_matrix)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    related_urls = [(url_list[i], similarities[i]) for i in sorted_indices if i != target_idx][:top_n]
    return related_urls

# Streamlit UI
st.title("🔗 Internal Linking Helper")

sitemap_url = st.text_input("Enter Sitemap URL:", "https://example.com/sitemap.xml")
if st.button("Process Sitemap"):
    st.write("Extracting URLs...")
    urls = extract_sitemap_urls(sitemap_url)
    
    st.write(f"Found {len(urls)} URLs. Generating embeddings...")
    embeddings = generate_embeddings(urls)
    
    url_list = list(embeddings.keys())
    embedding_matrix = np.array(list(embeddings.values()))
    
    st.success("Embeddings generated! Select a URL to find related pages.")
    selected_url = st.selectbox("Select a URL:", url_list)
    
    if st.button("Find Related Pages"):
        related_pages = find_related_pages(selected_url, url_list, embedding_matrix)
        st.write("### Related Pages for Internal Linking:")
        for url, score in related_pages:
            st.write(f"🔗 [{url}]({url}) (Similarity: {score:.2f})")
