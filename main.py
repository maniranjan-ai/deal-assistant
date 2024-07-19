import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.rfp_helper import *
import fitz  # PyMuPDF
import time


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


def load_chatbot(filename=None):

    def display_text_in_chunks(text, delay=0.02):
        chat_output = st.empty()
        for i in range(1, len(text) + 1):
            current_text = text[:i]
            chat_output.markdown(f"**Bot:** {current_text}")
            time.sleep(delay)

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
            <h2>Deal Assistant Chatbot</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if 'current_file' not in st.session_state or st.session_state.current_file == '':
        st.error("No file selected.")
        # st.button("Navigate to file browser", on_click=navigate_action)
        return 
    elif 'current_file' in st.session_state:
        filename = st.session_state.current_file
        print("file name is", filename)
        knowledgeBase = load_knowledge_base(filename=filename)
        llm = load_llm()
        prompt = load_prompt()

        # Create a form for user input
        with st.form(key='user_input_form', clear_on_submit=True):
            user_input = st.text_input("You:", key="input_text")
            submit_button = st.form_submit_button(label="Send")

        if submit_button:
            if user_input:
                similar_embeddings = knowledgeBase.similarity_search(user_input)
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

                response = rag_chain.invoke(user_input)
                st.session_state.conversation.append(("You", user_input))
                st.session_state.conversation.append(("Bot", ""))
                # Display streaming response
                display_text_in_chunks(response)

                # Update conversation with the complete response
                st.session_state.conversation[-1] = ("Bot", response)

    # Display conversation
        if st.session_state.conversation:
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for speaker, text in reversed(st.session_state.conversation):
                css_class = "user" if speaker == "You" else "bot"
                st.markdown(
                    f"<div class='chat-message {css_class}'><strong>{speaker}:</strong> {text}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)


# Define button actions
def continue_action(uploaded_file):
    global current_file
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


if __name__ == '__main__':
    st.title("Deal Assistant")

    if 'navigation' not in st.session_state:
        st.session_state.navigation = "File Browser"

    st.sidebar.title("Menu")
    options = ["File Browser", "Deal Assistant Bot", "About Deal Assistant"]
    selection = st.sidebar.selectbox("Go to", options, index=options.index(st.session_state.navigation))


    st.session_state.navigation = selection

    if st.session_state.navigation == "File Browser":
        load_rfp_selector()

    elif st.session_state.navigation == "Deal Assistant Bot":
        load_chatbot()

    elif st.session_state.navigation == "About Deal Assistant":
        st.header("About")
        st.write("""
            **Deal Assistant** is an application designed to assist users in handling and interacting with Request for Proposal (RFP) documents and engaging in chatbot interactions. Below is an overview of its core features and functionality:

            ### Core Features:

            **1. RFP File Selection:**
            - **File Upload**: Users can upload PDF files containing RFP documents.
            - **File Viewing**: Once uploaded, the content of the RFP document is extracted for user review.
            - **File Management**: Users can select from a list of previously uploaded files to view their content.

            **2. Chatbot Interaction:**
            - **User Interaction**: Users can interact with a chatbot by asking questions. The chatbot utilizes the  RFP data to generate responses based on the user input.
            - **Conversation History**: The chatbot maintains a history of interactions for context and reference.

            **3. Navigation:**
            - **Page Navigation**: Users can navigate between the RFP File Selector and the Chatbot interface through a sidebar menu.

            **4. State Management:**
            - **Session State**: Uses session state to manage the current file, navigation state, and refresh functionality.

            ### Usage Context:

            The "Deal Assistant" application is useful in scenarios where users need to manage and analyze RFP documents, interact with a chatbot for queries related to the content of those documents, and maintain a record of their interactions with the chatbot. It is likely intended for professionals who deal with RFPs and need an efficient tool for document management and interactive assistance.
            """)
