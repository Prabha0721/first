import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd

st.header("Extract URLs from Sitemap")
user_input = st.text_input("Enter sitemap URL:")

if user_input:
    try:
        # Make a request to the sitemap URL
        page = requests.get(user_input)
        soup = BeautifulSoup(page.content, 'xml')  # Change parser to 'xml' for sitemap
        urls = soup.find_all('loc')

        data = []

        for url in urls:
            currentURL = url.get_text()
            if '.pdf' not in currentURL:  # Exclude PDF links
                try:
                    # Make a request to the current URL
                    page = requests.get(currentURL)
                    soup = BeautifulSoup(page.content, 'html.parser')

                    title = soup.find('title')
                    title_text = title.get_text() if title else "Title not found"

                    description = soup.find('meta', attrs={'name': 'description'})
                    description_text = description['content'] if description else "Description not found"

                    canonical = soup.find('link', rel='canonical')
                    canonical_url = canonical['href'] if canonical else "Canonical not found"

                    h1 = soup.find('h1')
                    h1_text = h1.get_text() if h1 else "H1 not found"

                    data.append({
                        'URL': currentURL,
                        'Title': title_text,
                        'Description': description_text,
                        'H1': h1_text,
                        'Canonical URL': canonical_url
                    })
                except Exception as e:
                    st.error(f"Error processing URL {currentURL}: {e}")

        # Create a DataFrame and display it
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df)  # Use st.dataframe to display the DataFrame
        else:
            st.warning("No valid URLs found.")
    except Exception as e:
        st.error(f"Error fetching sitemap: {e}")
