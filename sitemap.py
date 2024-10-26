!pip install advertools
import streamlit as st
st.header("To Extract Urls")

user_input = st.text_input("Enter sitemap url:")
st.write(user_input)

import advertools as adv
import pandas as pd

sitemap = adv.sitemap_to_df(user_input)
sitemap.to_csv("output.csv")
