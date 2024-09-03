# import os
#
# the_key=os.getenv("OPENAI_API_KEY")
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables from .env file
load_dotenv()

the_key = st.secrets["OPENAI_API_KEY"]
da_pass = st.secrets["da_pass"]