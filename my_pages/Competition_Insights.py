import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_option_menu import option_menu
from utils.rfp_helper import *
import fitz  # PyMuPDF
import time
from streamlit_feedback import streamlit_feedback
from datetime import datetime, timedelta
import pandas as pd
from utils.ci_helper import *


def save_chat_history_to_excel(username):
    df = pd.DataFrame(st.session_state.chat_history)
    df.to_excel(f"{username}_chat_history.xlsx", index=False)


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

        if 'current_file_ci' not in st.session_state or st.session_state.current_file_ci == '':
            st.error("No file selected.")
            # st.button("Navigate to file browser", on_click=navigate_action)
            return
        elif 'current_file_ci' in st.session_state and 'current_file' in st.session_state:
            filename_ci = st.session_state.current_file_ci
            filename = st.session_state.current_file
            print("file name is", filename)
            print("competitor file name is", filename_ci)

            kb1 = load_knowledge_base("vectorstore/"+filename)
            kb2 = load_knowledge_base("vectorstore_ci/"+filename_ci)
            llm = load_llm()

            # Create a form for user input
            st.session_state.messages.append({"role": "user", "content": input_message})

            # Perform similarity search on both knowledge bases
            similar_docs_kb1 = kb1.similarity_search(input_message, k=3)
            similar_docs_kb2 = kb2.similarity_search(input_message, k=3)

            # Retrieve the context information
            RFP_context = format_docs(similar_docs_kb1)  # Assuming rfp_retriever is a function
            competitors_context = format_docs(similar_docs_kb2)  # Assuming compt_retriever is a function

            # Generate and print the prompt
            prompt_string = create_and_print_prompt({
                "RFP_context": RFP_context,
                "competitors_context": competitors_context,
                "question": input_message
            })

            # Pass the prompt to the LLM
            llm_response = llm(prompt_string)

            # Parse the LLM response
            response = StrOutputParser().parse(llm_response.content)

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
            # print(st.session_state.chat_history)

        with st.form('form'):
            streamlit_feedback(feedback_type="thumbs", align="flex-start", key='fb_k')
            st.form_submit_button('Save feedback', on_click=fbcb)


def continue_action(uploaded_file):
    DB_FAISS_PATH = 'vectorstore_ci/' + uploaded_file.name
    loader = PyPDFLoader("uploaded_files_ci/" + uploaded_file.name)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=the_key))
    vectorstore.save_local(DB_FAISS_PATH)
    st.session_state.current_file_ci = uploaded_file.name
    st.session_state.navigation = "ChatBot"


def continue_action_full_context(uploaded_file):
    global current_file
    text = extract_text_from_pdf("uploaded_files_ci/" + uploaded_file.name)
    output_txt_path = 'txtstore_ci/' + uploaded_file.name.replace(".pdf", "") + ".txt"
    with open(output_txt_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    st.session_state.current_file_ci = uploaded_file.name
    st.session_state.navigation = "ChatBot"

def continue_action_saved_file():
    st.session_state.navigation = "ChatBot"

def load_chatbot():
    suggestions = [
        "Share the list of the competitive products available with Competitor B, in response to this RFP?",
        "Show me the discount trends for data insights offered by Competitor B in the last year?",
        "What are the strengths and weaknesses of Competitor B?",
        "What is the USP of the competitors product - Data Insight Analytics and what is the launch roadmap for the product?"]

    # Display suggestions as buttons
    rows = len(suggestions) // 2 + (len(suggestions) % 2 > 0)  # Calculate number of rows needed
    key_counter = 0

    for row in range(rows):
        cols = st.columns(2)  # Create 2 columns
        for col in range(2):
            index = row * 2 + col  # Calculate the correct index for suggestions
            if index < len(suggestions):  # Check if index is within bounds
                if cols[col].button(suggestions[index], key=f"button_{index}", use_container_width=True):
                    handle_input(suggestions[index], "your_openai_api_key")

    with st.spinner('Scanning document.Please wait...'):
        if prompt := st.chat_input():
            input_message = prompt
            # st.session_state.current_file = "competitors_sample_data1.pdf"
            if 'current_file_ci' not in st.session_state or st.session_state.current_file_ci == '':
                st.error("No file selected.")
                # st.button("Navigate to file browser", on_click=navigate_action)
                return
            elif 'current_file_ci' in st.session_state:
                filename = st.session_state.current_file_ci
                print("file name is", filename)
                handle_input(input_message, "your_openai_api_key")

def load_file_browser():
    # Directory to save uploaded files
    UPLOAD_DIR = 'uploaded_files_ci'

    if 'current_file_ci' not in st.session_state:
        st.session_state.current_file_ci = ''

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

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
    selected_file = ""
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
        st.session_state.current_file_ci = selected_file

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


def render():
    with st.sidebar:
        openai_api_key = the_key

    options_ci = ["Competitor File Browser", "ChatBot"]

    if 'navigation' not in st.session_state or st.session_state.navigation not in options_ci:
        st.session_state.navigation = "Competitor File Browser"

    if "current_file_ci" not in st.session_state:
        st.session_state.current_file_ci = ""

    with st.sidebar:
        selection = option_menu(
            menu_title="",
            options=options_ci,
            icons=[],
            menu_icon=["heart-eyes-fill"],
            default_index=options_ci.index(st.session_state.navigation),
        )

    st.session_state.navigation = selection

    if st.session_state.navigation == "Competitor File Browser":
        load_file_browser()

    elif st.session_state.navigation == "ChatBot":
        load_chatbot()