from langchain.prompts import ChatPromptTemplate
from utils.key_data import *
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_knowledge_base(filename):
    embeddings = OpenAIEmbeddings(api_key=the_key)
    DB_FAISS_PATH = 'vectorstore/'+ filename
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db


# function to load the OPENAI LLM
def load_llm():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                     api_key=the_key)
    return llm


# creating prompt template using langchain
def load_prompt():
    prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf, answer "Not Sure! I may not be the right one to answer that!"
         """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)