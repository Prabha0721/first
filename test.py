import streamlit as st
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from io import StringIO

# Load the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_sitemap_urls(sitemap_url):
    """Fetch and parse URLs from sitemap.xml"""
    response = requests.get(sitemap_url)
    root = ET.fromstring(response.content)
    urls = [elem.text for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    return urls

def get_text_from_url(url):
    """Scrape and extract visible text from a webpage (consider all content)"""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove non-visible elements (script, style, footer, header, etc.)
        for element in soup(["script", "style", "footer", "header", "nav", "aside"]):
            element.decompose()
        
        # Extract text from all remaining tags
        text = " ".join([element.get_text() for element in soup.find_all(True)])  # True matches all tags
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

def find_related_pages(target_url, url_list, embedding_matrix, top_n=50):
    """Find top-N related pages based on cosine similarity"""
    target_idx = url_list.index(target_url)
    similarities = cosine_similarity([embedding_matrix[target_idx]], embedding_matrix)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    related_urls = [(url_list[i], similarities[i]) for i in sorted_indices if i != target_idx][:top_n]
    return related_urls

def create_csv(target_url, related_urls):
    """Creates a CSV from target URL and related URLs"""
    data = {
        "Target URL": [target_url] * len(related_urls),
        "Related URL": [url for url, _ in related_urls],
        "Similarity Score": [score for _, score in related_urls]
    }
    df = pd.DataFrame(data)
    
    # Convert the DataFrame to CSV
    csv = df.to_csv(index=False)
    return csv

# Streamlit UI
st.title("🔗 Internal Linking Helper")

# Sitemap URL input
sitemap_url = st.text_input("Enter Sitemap URL:")

# Process sitemap button
if st.button("Process Sitemap"):
    # Extract URLs from sitemap
    st.write("Extracting URLs...")
    urls = extract_sitemap_urls(sitemap_url)
    
    st.write(f"Found {len(urls)} URLs. Generating embeddings...")
    
    # Generate embeddings for each URL
    embeddings = generate_embeddings(urls)
    
    # Prepare lists for URL and corresponding embeddings
    url_list = list(embeddings.keys())
    embedding_matrix = np.array(list(embeddings.values()))
    
    # Store the processed state so the user can select a URL later
    st.session_state.urls = url_list
    st.session_state.embedding_matrix = embedding_matrix
    st.session_state.embeddings = embeddings
    
    st.success("Embeddings generated! Now, type a URL to find related pages.")

# URL input to search related pages
if "urls" in st.session_state:
    selected_url = st.text_input("Type a URL to find related pages:", "")
    
    if selected_url:
        # Validate that the entered URL exists in the sitemap
        if selected_url in st.session_state.urls:
            # Button to trigger finding related pages
            if st.button("Find Related Pages"):
                # Find the related pages based on cosine similarity
                related_pages = find_related_pages(selected_url, st.session_state.urls, st.session_state.embedding_matrix)
                st.write("### Related Pages for Internal Linking:")
                for url, score in related_pages:
                    st.write(f"🔗 [{url}]({url}) (Similarity: {score:.2f})")

                # Create CSV from target URL and related URLs
                csv_data = create_csv(selected_url, related_pages)

                # Create a download button for the CSV file
                st.download_button(
                    label="Download Related Pages as CSV",
                    data=csv_data,
                    file_name=f"related_pages_{selected_url.replace('https://', '').replace('/', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.write(f"URL `{selected_url}` is not in the sitemap. Please check and try again.")
else:
    st.info("Please process the sitemap first.")
