import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd

st.header("Extract elements from Sitemap")
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
                    
                    if canonical_url == currentURL:
                        canonical_url = "Self-Referential"

                    robots=soup.find('meta', attrs={'name':'robots'})
                    robots_text=robots['content'] if robots else "Not found"

                    h1 = soup.find('h1')
                    h1_text = h1.get_text() if h1 else "H1 not found"

                    data.append({
                        'URL': currentURL,
                        'Title': title_text,
                        'Description': description_text,
                        'H1': h1_text,
                        'Canonical URL': canonical_url,
                        'Noindex/Nofollow':robots_text
                    })
                except Exception as e:
                    st.error(f"Error processing URL {currentURL}: {e}")

        # Create a DataFrame and display it
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df)  # Display the DataFrame

            # Add a download button for the CSV file
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='sitemap_data.csv',
                mime='text/csv'
            )
        else:
            st.warning("No valid URLs found.")
    except Exception as e:
        st.error(f"Error fetching sitemap: {e}")
