import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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

        if 'current_file' not in st.session_state or st.session_state.current_file == '':
            st.error("No file selected.")
            # st.button("Navigate to file browser", on_click=navigate_action)
            return
        elif 'current_file' in st.session_state:
            filename = st.session_state.current_file
            print("file name is", filename)
            knowledgeBase = load_knowledge_base(filename=filename)
            kb1 = load_knowledge_base(filename)
            kb2 = load_knowledge_base("competitors_sample_data1.pdf")
            llm = load_llm()

            # Create a form for user input
            st.session_state.messages.append({"role": "user", "content": input_message})

            # similar_embeddings = knowledgeBase.similarity_search(input_message, k=10)
            # similar_embeddings = FAISS.from_documents(documents=similar_embeddings,
            #                                           embedding=OpenAIEmbeddings(
            #                                               api_key=the_key))

            # Perform similarity search on both knowledge bases
            similar_docs_kb1 = kb1.similarity_search(input_message, k=3)
            similar_docs_kb2 = kb2.similarity_search(input_message, k=3)

            # Combine the results from both searches
            combined_docs = similar_docs_kb1 + similar_docs_kb2

            # Optional: Shuffle or simply take the top N if you cannot determine scores
            # In this example, we're just selecting the first 10 from the combined list
            top_combined_results = combined_docs
            # for doc in top_combined_results:
            #     print(doc.page_content)

            # creating the chain for integrating llm,prompt,stroutputparser
            # retriever = similar_embeddings.as_retriever()
            rfp_retriever = FAISS.from_documents(documents=similar_docs_kb1,
                                             embedding=OpenAIEmbeddings(api_key=the_key)).as_retriever()
            compt_retriever = FAISS.from_documents(documents=similar_docs_kb2,
                                             embedding=OpenAIEmbeddings(api_key=the_key)).as_retriever()
            # retriever = similar_embeddings.as_retriever()

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


def render():
    with st.sidebar:
        openai_api_key = the_key

    # st.title("💬 Chatbot")
    # st.caption("🚀 A Streamlit chatbot powered by OpenAI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

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
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
            # st.session_state.current_file = "competitors_sample_data1.pdf"
            if 'current_file' not in st.session_state or st.session_state.current_file == '':
                st.error("No file selected.")
                # st.button("Navigate to file browser", on_click=navigate_action)
                return
            elif 'current_file' in st.session_state:
                filename = st.session_state.current_file
                print("file name is", filename)
                handle_input(input_message, "your_openai_api_key")
