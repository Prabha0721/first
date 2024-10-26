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
        else:
            st.warning("No URLs found in the sitemap.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
