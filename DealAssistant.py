from openai import OpenAI
import streamlit as st
import pandas as pd
import hashlib

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")


st.title("Deal Assistant")
st.caption("🚀 How can we help you today!")


def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if "current_file" not in st.session_state:
        st.session_state.current_file = ""


initialize_session_state()

users_db = {
    "user1": {"password": hashlib.sha256("password1".encode()).hexdigest()},
    "user2": {"password": hashlib.sha256("password2".encode()).hexdigest()},
}

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_chat_history_from_excel(username):
    try:
        df = pd.read_excel(f"{username}_chat_history.xlsx")
        return df.to_dict(orient='records')
    except FileNotFoundError:
        return []

def login(username, password):
    if username in users_db and users_db[username]["password"] == hash_password(password):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.chat_history = load_chat_history_from_excel(username)
        return True
    else:
        st.error("Invalid username or password")
        return False

if not st.session_state.logged_in:
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login = login(username, password)
        if login:
            st.success("Logged in successfully")
            st.experimental_rerun()


