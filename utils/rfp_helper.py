from langchain.prompts import ChatPromptTemplate
from utils.key_data import *
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai
import fitz
from openai import OpenAI

client = OpenAI(api_key=the_key)


def load_knowledge_base(filename):
    embeddings = OpenAIEmbeddings(api_key=the_key, model='text-embedding-ada-002')
    DB_FAISS_PATH = filename
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db


# function to load the OPENAI LLM
def load_llm():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4,
                     api_key=the_key)
    return llm


def load_prompt():
    prompt = """
    You are an intelligent assistant with access to a specific Request for Proposal (RFP) document. 
    Your task is to answer user queries strictly based on the provided context from the RFP.

    Instructions:
    1. Carefully read the provided context from the RFP and competitor data.
    2. Answer the user query using only the information available in the context.
    3. If the context does not contain the necessary information to answer the query, respond with: "Not Sure! I may not be the right one to answer that!"
    4. Your response should be clear, concise, and relevant to the user's question.

    Context: 
    Request for Proposal (RFP) Information: {context}

    User Query:  {question}

    Response:
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def load_prompt_ci(RFP_context, competitors_context, question):
    # Create the prompt template as a string
    prompt_string = f"""
    You are an intelligent assistant with access to a specific Request for Proposal (RFP) document and detailed competitor information. 
    Your task is to answer user queries strictly based on the provided context from the RFP and competitor data.

    Instructions:
    1. Carefully read the provided context from the RFP and competitor data.
    2. Answer the user query using only the information available in the context.
    3. If the context does not contain the necessary information to answer the query, respond with: "Not Sure! I may not be the right one to answer that!"
    4. Your response should be clear, concise, and relevant to the user's question.

    Context: 
    Request for Proposal (RFP) Information: {RFP_context}
    Competitor Information: {competitors_context}

    User Query: 
    {question}

    Response:
    """
    # Print the string version of the prompt
    print("Generated Prompt String:\n", prompt_string)

    # Convert the string prompt to a ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(prompt_string)

    return prompt_template


def create_and_print_prompt(inputs):
    RFP_context = inputs["RFP_context"]
    competitors_context = inputs["competitors_context"]
    question = inputs["question"]

    # Generate the prompt string using the provided contexts and question
    prompt_string = f"""
    You are an intelligent assistant with access to a specific Request for Proposal (RFP) document and detailed competitor information. 
    Your task is to answer user queries strictly based on the provided context from the RFP and competitor data.

    Instructions:
    Review Context: Read the provided RFP and competitor information carefully.
    Answer Queries: Use only the information available in the context to respond.
    Unanswered Queries: If the context does not have the required information, respond with: "Not Sure! I may not be the right one to answer that!"
    Response Style: Ensure your answers are clear, concise, and relevant.

    Context: 
    Request for Proposal (RFP) Information: {RFP_context}
    Competitor Information: {competitors_context}

    User Query: 
    {question}

    Response:
    """

    # Print the prompt string
    print("Generated Prompt String:\n", prompt_string)

    # Return the prompt string
    return prompt_string


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ask_gpt4_full_context(question, context):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are an assistant tasked with answering questions based on the content of a given RFP (Request for Proposal) document."},
            {"role": "user", "content": f"""
    You are an assistant tasked with answering questions based on the content of a given RFP (Request for Proposal) document. Your responses should strictly adhere to the information provided in the context of the PDF content.

    Instructions:
    1. Carefully read the provided context from the RFP.
    2. Answer the question based on the context.
    3. If the context does not contain the answer, respond with: "Not Sure! I may not be the right one to answer that!"

    Here are the details:
    Context: {context}
    Question: {question}

    Please provide your response below:
    """}
        ]
    )
    return response.choices[0].message.content


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def load_full_knowledge_base(filename):
    txt_path = 'txtstore/' + filename[:-4] + ".txt"
    with open(txt_path, "r", encoding="utf-8") as text_file:
        text = text_file.read()
    return text
