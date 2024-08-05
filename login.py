import streamlit as st
from app import add_footer


def login():
    logo_url = "static/ACN.svg"  # Replace with the URL or path to your logo
    st.image(logo_url, width=50, use_column_width=False)

    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state['logged_in'] = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

    add_footer()
