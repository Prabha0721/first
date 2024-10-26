import streamlit as st
import advertools as adv
import pandas as pd

st.header("Extract URLs from Sitemap")
user_input = st.text_input("Enter sitemap URL:")

if user_input:
    try:
        # Extract URLs from the sitemap
        sitemap_df = adv.sitemap_to_df(user_input)

        # Display the DataFrame
        if not sitemap_df.empty:
            st.success("Sitemap extracted successfully!")
            st.dataframe(sitemap_df)
            
            # Create a download button
            csv = sitemap_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='sitemap_urls.csv',
                mime='text/csv',
            )
        else:
            st.warning("No URLs found in the sitemap.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
