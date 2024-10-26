import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd

myURL = 'https://www.jllhomes.co.in/sitemap.xml?page=sitemap'
page = requests.get(myURL)
soup = BeautifulSoup(page.content, 'html.parser')
urls = soup.find_all('loc')

data = []

for url in urls:
    currentURL = url.get_text()
    if '.pdf' not in currentURL:
        page = requests.get(currentURL)
        soup = BeautifulSoup(page.content, 'html.parser')

        title = soup.find('title')
        title_text = title.get_text() if title else "Title not found"

        description=soup.find('meta', attrs={'name':'description'})
        description_text=description['content'] if description else "description not found"

        canonical = soup.find('link', rel='canonical')
        canonical_url = canonical['href'] if canonical else "Canonical not found"

        h1 = soup.find('h1')
        h1_text = h1.get_text() if h1 else "H1 not found"

        data.append({'URL': currentURL, 'Title': title_text, 'Description': description_text, 'H1': h1_text, 'Canonical URL': canonical_url})

df = pd.DataFrame(data)
display(df)
