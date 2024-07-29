from langchain.prompts import ChatPromptTemplate
from utils.key_data import *
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_knowledge_base(filename):
    embeddings = OpenAIEmbeddings(api_key=the_key, model = 'text-embedding-ada-002')
    DB_FAISS_PATH = 'vectorstore/'+ filename
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db


# function to load the OPENAI LLM
def load_llm():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0,
                     api_key=the_key)
    return llm


# creating prompt template using langchain
# def load_prompt():
#     prompt = """ You need to answer the question in the sentence as same as in the  pdf content. .
#         Given below is the context and question of the user.
#         context = {context}
#         question = {question}
#         if the answer is not in the pdf, answer "Not Sure! I may not be the right one to answer that!"
#          """
#     prompt = ChatPromptTemplate.from_template(prompt)
#     return prompt

def load_prompt():
    prompt = """
    You are an assistant tasked with answering questions based on the content of a given RFP (Request for Proposal) document. Your responses should strictly adhere to the information provided in the context of the PDF content.

    Instructions:
    1. Carefully read the provided context from the RFP.
    2. Answer the question based on the context.
    3. If the context does not contain the answer, respond with: "Not Sure! I may not be the right one to answer that!"

    Here are the details:
    Context: {context}
    Question: {question}

    Please provide your response below:
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)