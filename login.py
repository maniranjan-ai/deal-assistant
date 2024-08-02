import streamlit as st


def login():
    # logo_url = "C:/Users/s.ck.srivastava/PycharmProjects/deal-assistant/static/ACN.svg"  # Replace with the URL or path to your logo
    # st.image(logo_url, width=150, use_column_width=False)

    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state['logged_in'] = True
        else:
            st.error("Invalid username or password")
