import streamlit as st
from streamlit_option_menu import option_menu
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.rfp_helper import *
import fitz  # PyMuPDF
import time
from streamlit_feedback import streamlit_feedback
from datetime import datetime, timedelta
import pandas as pd
import hashlib
from app import add_footer



# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    pdf_document = fitz.open(file_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    return text


def load_rfp_selector():
    # Directory to save uploaded files
    UPLOAD_DIR = 'uploaded_files'
    STATIC_DIR = 'static'
    global current_file

    # Logout button at the top of the main page
    # Logout button at the top of the main page
    # col1, col2, col3 = st.columns([3,5,2])
    # if col3.button('Logout'):
    #     logout()
    #     st.experimental_rerun()

    if 'current_file' not in st.session_state:
        st.session_state.current_file = ''

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # Streamlit app
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');

        .main {
            background-color: #e6eef1;
            font-family: 'Roboto', sans-serif;
        }
        .stTextInput > div > div > input {
            border: 2px solid #4a4a4a;
            border-radius: 8px;
            padding: 10px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #0073e6;
            box-shadow: 0 0 10px rgba(0, 115, 230, 0.5);
        }
        .stButton > button {
            background-color: #0073e6;
            color: white;
            border: 2px solid #0073e6;
            border-radius: 8px;
            padding: 8px 16px;
            margin-top: 10px;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #005bb5;
            border-color: #005bb5;
        }
        .stFileUploader > label > div {
            color: #0073e6;
        }
        .css-1ekf1nj {
            background-color: white;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .chat-message {
            background-color: #fff;
            border-radius: 8px;
            padding: 10px 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .chat-message.user {
            background-color: #0073e6;
            color: white;
            align-self: flex-end;
        }
        .chat-message.bot {
            background-color: #f0f2f6;
            color: black;
            align-self: flex-start;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            height: 50px;
            margin-right: 15px;
        }
        .file-dropdown {
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div>
            <h2>File Browser</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    # Layout: File uploader and dropdown menu side by side with OR
    col1, col2, col3 = st.columns([2, 0.3, 1.3])

    # File uploader widget
    with col1:
        uploaded_file = st.file_uploader("Upload document", type=["pdf"])

    # OR text
    with col2:
        st.markdown("<h5 style='text-align: center; margin-top: 60px;'>OR</h5>", unsafe_allow_html=True)

    # Dropdown menu for uploaded files
    with col3:
        uploaded_files = os.listdir(UPLOAD_DIR)
        if uploaded_files:
            selected_file = st.selectbox(
                "Select file to view",
                options=[""] + uploaded_files,
                key="file_dropdown",
                index=0,
            )
    if selected_file and selected_file != "":
        file_path = os.path.join(UPLOAD_DIR, selected_file)
        pdf_text = extract_text_from_pdf(file_path)
        st.text_area("File Content", pdf_text, height=300, key="dropdown_file")
        st.session_state.current_file = selected_file

        # Display the result based on which button was clicked
        st.button("Continue", on_click=continue_action_saved_file)

    # Handle file upload
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}

        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        pdf_text = extract_text_from_pdf(file_path)
        st.text_area("File Content", pdf_text, height=300, key="upload text area")
        st.session_state.uploaded_files.append(uploaded_file.name)

        # Display the result based on which button was clicked
        st.button("Continue", on_click=continue_action(uploaded_file))


# Function to filter messages based on time period
def filter_messages(time_period):
    now = datetime.now()
    if time_period == "last 7 days":
        return [msg for msg in st.session_state.chat_history if now - msg["timestamp"] <= timedelta(days=7)]
    elif time_period == "last 30 days":
        return [msg for msg in st.session_state.chat_history if now - msg["timestamp"] <= timedelta(days=30)]
    else:
        return st.session_state.chat_history


# Function to handle user input
def handle_input(user_input, openai_api_key):

    with st.spinner('Scanning document.Please wait...'):
        input_message = user_input
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        if 'current_file' not in st.session_state or st.session_state.current_file == '':
            st.error("No file selected.")
            # st.button("Navigate to file browser", on_click=navigate_action)
            return
        elif 'current_file' in st.session_state:
            filename = st.session_state.current_file
            print("file name is", filename)
            try:
                knowledgeBase = load_full_knowledge_base(filename)
                response = ask_gpt4_full_context(input_message, knowledgeBase)
                print("Used full context for response")
            except:
                print("Using RAG for response")
                knowledgeBase = load_knowledge_base(filename=filename)
                llm = load_llm()
                prompt = load_prompt()

                # Create a form for user input
                st.session_state.messages.append({"role": "user", "content": input_message})

                similar_embeddings = knowledgeBase.similarity_search(input_message)
                similar_embeddings = FAISS.from_documents(documents=similar_embeddings,
                                                          embedding=OpenAIEmbeddings(
                                                              api_key=the_key))

                # creating the chain for integrating llm,prompt,stroutputparser
                retriever = similar_embeddings.as_retriever()
                rag_chain = (
                        {"context": retriever | format_docs,
                         "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                )

                response = rag_chain.invoke(input_message)
        st.chat_message("user").write(input_message)
        st.session_state.messages.append({"role": "assistant", "content": response})

        def stream_data():
            for word in response.split(" "):
                yield word + " "
                time.sleep(0.2)

        st.chat_message("assistant").write_stream(stream_data)
        message_id = len(st.session_state.chat_history)

        st.session_state.chat_history.append({
            "question": input_message,
            "answer": response,
            "message_id": message_id,
            "timestamp": datetime.now()
        })

        save_chat_history_to_excel(st.session_state.username)

        def fbcb():
            message_id = len(st.session_state.chat_history) - 1
            if message_id >= 0:
                st.session_state.chat_history[message_id]["feedback"] = st.session_state.fb_k
            print(st.session_state.chat_history)

        with st.form('form'):
            streamlit_feedback(feedback_type="thumbs", align="flex-start", key='fb_k')
            st.form_submit_button('Save feedback', on_click=fbcb)
def load_chatbot():
    # CSS for the logout button
    # Logout button at the top of the main page

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');

        .main {
            background-color: #e6eef1;
            font-family: 'Roboto', sans-serif;
        }
        .stTextInput > div > div > input {
            border: 2px solid #4a4a4a;
            border-radius: 8px;
            padding: 10px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #0073e6;
            box-shadow: 0 0 10px rgba(0, 115, 230, 0.5);
        }
        .stButton > button {
            background-color: #0073e6;
            color: white;
            border: 2px solid #0073e6;
            border-radius: 8px;
            padding: 8px 16px;
            margin-top: 10px;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #005bb5;
            border-color: #005bb5;
        }
        .stFileUploader > label > div {
            color: #0073e6;
        }
        .css-1ekf1nj {
            background-color: white;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .chat-message {
            background-color: #fff;
            border-radius: 8px;
            padding: 10px 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .chat-message.user {
            background-color: #0073e6;
            color: white;
            align-self: flex-end;
        }
        .chat-message.bot {
            background-color: #f0f2f6;
            color: black;
            align-self: flex-start;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            height: 50px;
            margin-right: 15px;
        }
        .file-dropdown {
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # col1, col2, col3 = st.columns([3,5,2])
    # if col3.button('Logout'):
    #     logout()
    #     st.experimental_rerun()

    with st.sidebar:
        openai_api_key = the_key

        # if st.session_state.current_file:
        #     st.markdown(
        #         f"""
        #             <div style="position: absolute; top: 10px; right: 10px; background-color: lightgray; padding: 5px; border-radius: 5px;">
        #                {st.session_state.current_file}
        #             </div>
        #             """,
        #         unsafe_allow_html=True
        #     )


    st.title("💬 Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    #
    # col1, col2 = st.columns([1, 3])

    # Display chat history in the left column

    # with col2:

    # if 'current_file' in st.session_state:
    #     st.chat_message("assistant").write("Context is set with " + st.session_state.current_file)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    suggestions = ["Please share a Summary for the RFP", "Please share the submission guidelines for this RFP?", "Please highlight the important dates for this RFP", "Show me the detail budget and funding for this RFP?"]

    # Display suggestions as buttons
    rows = len(suggestions) // 2 + (len(suggestions) % 2 > 0)  # Calculate number of rows needed
    key_counter = 0

    for row in range(rows):
        cols = st.columns(2)  # Create 2 columns
        for col in range(2):
            index = row * 2 + col  # Calculate the correct index for suggestions
            if index < len(suggestions):  # Check if index is within bounds
                if cols[col].button(suggestions[index], key=f"button_{index}"):
                    handle_input(suggestions[index], "your_openai_api_key")


    with st.spinner('Scanning document.Please wait...'):
        if prompt := st.chat_input():
            input_message = prompt
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()

            if 'current_file' not in st.session_state or st.session_state.current_file == '':
                st.error("No file selected.")
                # st.button("Navigate to file browser", on_click=navigate_action)
                return
            elif 'current_file' in st.session_state:
                filename = st.session_state.current_file
                print("file name is", filename)
                try:
                    knowledgeBase = load_full_knowledge_base(filename)
                    response = ask_gpt4_full_context(input_message,knowledgeBase)
                    print("Used full context for response")
                except:
                    print("Using RAG for response")
                    knowledgeBase = load_knowledge_base(filename=filename)
                    llm = load_llm()
                    prompt = load_prompt()

                    # Create a form for user input
                    st.session_state.messages.append({"role": "user", "content": input_message})

                    similar_embeddings = knowledgeBase.similarity_search(input_message, k=10)
                    similar_embeddings = FAISS.from_documents(documents=similar_embeddings,
                                                              embedding=OpenAIEmbeddings(
                                                                  api_key=the_key))

                    # creating the chain for integrating llm,prompt,stroutputparser
                    retriever = similar_embeddings.as_retriever()
                    rag_chain = (
                            {"context": retriever | format_docs,
                             "question": RunnablePassthrough()}
                            | prompt
                            | llm
                            | StrOutputParser()
                    )

                    response = rag_chain.invoke(input_message)
            st.chat_message("user").write(input_message)
            st.session_state.messages.append({"role": "assistant", "content": response})

            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.2)

            st.chat_message("assistant").write_stream(stream_data)
            message_id = len(st.session_state.chat_history)

            st.session_state.chat_history.append({
                "question": input_message,
                "answer": response,
                "message_id": message_id,
                "timestamp": datetime.now()
            })

            save_chat_history_to_excel(st.session_state.username)

            def fbcb():
                message_id = len(st.session_state.chat_history) - 1
                if message_id >= 0:
                    st.session_state.chat_history[message_id]["feedback"] = st.session_state.fb_k
                print(st.session_state.chat_history)

            with st.form('form'):
                streamlit_feedback(feedback_type="thumbs", align="flex-start", key='fb_k')
                st.form_submit_button('Save feedback', on_click=fbcb)


# Define button actions
# def continue_action(uploaded_file):
#     global current_file
#     DB_FAISS_PATH = 'vectorstore/' + uploaded_file.name
#     loader = PyPDFLoader("uploaded_files/" + uploaded_file.name)
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)
#     vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=the_key))
#     vectorstore.save_local(DB_FAISS_PATH)
#     st.session_state.current_file = uploaded_file.name
#     st.session_state.navigation = "Deal Assistant Bot"

def continue_action(uploaded_file):
    global current_file
    text = extract_text_from_pdf("uploaded_files/" + uploaded_file.name)
    output_txt_path = 'txtstore/' + uploaded_file.name.replace(".pdf","") +".txt"
    with open(output_txt_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)

    DB_FAISS_PATH = 'vectorstore/' + uploaded_file.name
    loader = PyPDFLoader("uploaded_files/" + uploaded_file.name)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=the_key))
    vectorstore.save_local(DB_FAISS_PATH)
    st.session_state.current_file = uploaded_file.name
    st.session_state.navigation = "Deal Assistant Bot"


def continue_action_saved_file():
    st.session_state.navigation = "Deal Assistant Bot"


def navigate_action():
    print("navigation called")
    st.session_state.navigation = "File Browser"


def save_chat_history_to_excel(username):
    df = pd.DataFrame(st.session_state.chat_history)
    df.to_excel(f"{username}_chat_history.xlsx", index=False)


# Function to load chat history from Excel
# Function to load chat history from Excel
def load_chat_history_from_excel(username):
    try:
        df = pd.read_excel(f"{username}_chat_history.xlsx")
        return df.to_dict(orient='records')
    except FileNotFoundError:
        return []

users_db = {
    "user1": {"password": hashlib.sha256("password1".encode()).hexdigest()},
    "user2": {"password": hashlib.sha256("password2".encode()).hexdigest()},
}

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def login(username, password):
    if username in users_db and users_db[username]["password"] == hash_password(password):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.chat_history = load_chat_history_from_excel(username)
        return True
    else:
        st.error("Invalid username or password")
        return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.chat_history = []


def render():

    st.title("Deal Assistant")


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

    # User login/logout interface
    # if not st.session_state.logged_in:
    #     st.header("Login")
    #     username = st.text_input("Username")
    #     password = st.text_input("Password", type="password")
    #     if st.button("Login"):
    #         login = login(username, password)
    #         if login:
    #             st.success("Logged in successfully")
    #             st.experimental_rerun()
    # else:
    if 'navigation' not in st.session_state:
        st.session_state.navigation = "File Browser"

    # st.sidebar.title("Menu")
    options = ["File Browser", "Deal Assistant Bot"]
    # selection = st.sidebar.selectbox("Go to", options, index=options.index(st.session_state.navigation))

    with st.sidebar:
        selection = option_menu(
            menu_title="",
            options=options,
            icons=[],
            menu_icon=["heart-eyes-fill"],
            default_index=options.index(st.session_state.navigation),
        )

    # st.session_state.chat_history = load_chat_history_from_excel()

    if st.session_state.current_file:
        with st.sidebar:
            st.write(st.session_state.current_file)

    with st.sidebar:
        st.header("Chat History")
        time_period = st.selectbox("Select time period", ["all time", "last 7 days", "last 30 days"])
        filtered_messages = filter_messages(time_period)
        user_role="👤"
        bot_role = "🤖"
        for chat in filtered_messages:
            st.write(f"{user_role}: {chat['question']}")
            st.write(f"{bot_role}: {chat['answer']}")

    st.session_state.navigation = selection

    if st.session_state.navigation == "File Browser":
        load_rfp_selector()

    elif st.session_state.navigation == "Deal Assistant Bot":
        load_chatbot()

    add_footer()

    # elif st.session_state.navigation == "About Deal Assistant":
    #     st.header("About")
    #     st.write("""
    #         **Deal Assistant** is an application designed to assist users in handling and interacting with Request for Proposal (RFP) documents and engaging in chatbot interactions. Below is an overview of its core features and functionality:
    #
    #         ### Core Features:
    #
    #         **1. RFP File Selection:**
    #         - **File Upload**: Users can upload PDF files containing RFP documents.
    #         - **File Viewing**: Once uploaded, the content of the RFP document is extracted for user review.
    #         - **File Management**: Users can select from a list of previously uploaded files to view their content.
    #
    #         **2. Chatbot Interaction:**
    #         - **User Interaction**: Users can interact with a chatbot by asking questions. The chatbot utilizes the  RFP data to generate responses based on the user input.
    #         - **Conversation History**: The chatbot maintains a history of interactions for context and reference.
    #
    #         **3. Navigation:**
    #         - **Page Navigation**: Users can navigate between the RFP File Selector and the Chatbot interface through a sidebar menu.
    #
    #         **4. State Management:**
    #         - **Session State**: Uses session state to manage the current file, navigation state, and refresh functionality.
    #
    #         ### Usage Context:
    #
    #         The "Deal Assistant" application is useful in scenarios where users need to manage and analyze RFP documents, interact with a chatbot for queries related to the content of those documents, and maintain a record of their interactions with the chatbot. It is likely intended for professionals who deal with RFPs and need an efficient tool for document management and interactive assistance.
    #         """)
