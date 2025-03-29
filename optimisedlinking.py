import asyncio
import aiohttp
import requests
import csv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import concurrent.futures
import time


# Initialize the model for embeddings generation
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Caching requests to avoid redundant network calls
requests_cache.install_cache('sitemap_cache', expire_after=3600)  # Cache expires in 1 hour


async def fetch_page(session, url):
    """Fetch the page content asynchronously."""
    try:
        async with session.get(url, timeout=5) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


async def get_page_links(session, url):
    """Extract all links from a page."""
    html_content = await fetch_page(session, url)
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
        links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
        return links
    return []


async def fetch_all_links(urls):
    """Fetch links from multiple pages concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [get_page_links(session, url) for url in urls]
        return await asyncio.gather(*tasks)


def extract_sitemap_urls(sitemap_url):
    """Extract URLs from the sitemap."""
    response = requests.get(sitemap_url)
    root = ET.fromstring(response.content)
    urls = [elem.text for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    return urls


def get_text_from_url(url):
    """Extract text content from a webpage."""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text.strip()
    except Exception as e:
        print(f"Error fetching text from {url}: {e}")
        return ""


def generate_embeddings_batch(urls):
    """Generate embeddings for multiple URLs in a batch."""
    texts = [get_text_from_url(url) for url in urls]
    embeddings = model.encode(texts, batch_size=32)
    return embeddings


def find_related_pages(target_url, url_list, embedding_matrix, top_n=3):
    """Find related pages based on cosine similarity."""
    target_idx = url_list.index(target_url)
    similarities = cosine_similarity([embedding_matrix[target_idx]], embedding_matrix)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    related_urls = [(url_list[i], similarities[i]) for i in sorted_indices if i != target_idx][:top_n]
    return related_urls


def save_to_csv(target_url, related_pages, filename="related_pages.csv"):
    """Save the related pages and their similarity scores to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Target URL', 'Related URL', 'Similarity Score'])
        for url, score in related_pages:
            writer.writerow([target_url, url, score])


def main():
    # User inputs
    sitemap_url = input("Enter Sitemap URL: ")
    target_url = input("Enter Target URL: ")

    # Step 1: Extract URLs from the sitemap
    print("Extracting URLs from sitemap...")
    urls = extract_sitemap_urls(sitemap_url)
    print(f"Found {len(urls)} URLs.")

    # Step 2: Fetch links from all pages asynchronously
    print("Fetching links from all pages...")
    start_time = time.time()
    links = asyncio.run(fetch_all_links(urls))
    print(f"Fetched links from {len(urls)} pages in {time.time() - start_time:.2f} seconds.")

    # Step 3: Generate embeddings for the pages in batches
    print("Generating embeddings...")
    batch_size = 32  # You can adjust this batch size based on memory and performance
    url_list = list(urls)
    embedding_matrix = np.array(generate_embeddings_batch(url_list))
    print("Embeddings generated.")

    # Step 4: Find related pages for the target URL
    print("Finding related pages...")
    related_pages = find_related_pages(target_url, url_list, embedding_matrix)
    
    # Display related pages
    if related_pages:
        print(f"\nRelated pages for {target_url}:")
        for url, score in related_pages:
            print(f"🔗 [{url}] ({score:.2f})")
        
        # Save the results to a CSV file
        save_to_csv(target_url, related_pages)
        print(f"\nResults saved to 'related_pages.csv'.")
    else:
        print(f"No related pages found for {target_url}.")


if __name__ == "__main__":
    main()
