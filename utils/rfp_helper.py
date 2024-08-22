from langchain.prompts import ChatPromptTemplate
from utils.key_data import *
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import fitz
from openai import OpenAI
from langchain.chains import LLMChain, RouterChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import *

client = OpenAI(api_key=the_key)


def load_knowledge_base(filename):
    embeddings = OpenAIEmbeddings(api_key=the_key, model='text-embedding-ada-002')
    DB_FAISS_PATH = 'vectorstore/' + filename
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


def load_summarization_prompt(RFP_context, question):
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

        User Query: 
        {question}

        Response:
        """

    return prompt_string


def load_chunk_summarization_prompt(RFP_context_chunk):
    # Create the prompt template as a string
    prompt_string = f"""
            Please summarize the following text in concise manner focusing on main points:
            {RFP_context_chunk}

            Response:
            """

    return prompt_string


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


# Define the router function using the LLM
def router_func(input_text):
    routing_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="""Decide whether the input prompt is asking for a summarization task or a general information retrieval task.
        If it is a summarization task, respond with 'summarization'. Otherwise, respond with 'general'.
        Input: {input_text}"""
    )
    routing_chain = LLMChain(llm=load_llm(), prompt=routing_prompt)
    task_type = routing_chain.run(input_text)

    return task_type


def split_text(text, max_tokens=1000):
    tokens = text.split()
    chunks = []
    while len(tokens) > 0:
        chunk = tokens[:max_tokens]
        chunks.append(" ".join(chunk))
        tokens = tokens[max_tokens:]
    return chunks


def recursive_summarization(text, summarization_chain, max_tokens=1000, min_length=100):
    # Step 1: Split the text into manageable chunks
    chunks = split_text(text, max_tokens=max_tokens)

    # Step 2: Summarize each chunk
    summaries = [summarization_chain.invoke(load_chunk_summarization_prompt(RFP_context_chunk=chunk)) for chunk in chunks]

    # Step 3: Combine summaries
    combined_summary = " ".join(summaries)

    # Step 4: If the combined summary is too long, recursively summarize it
    if len(combined_summary.split()) > min_length:
        return recursive_summarization(combined_summary, summarization_chain, max_tokens=max_tokens, min_length=min_length)

    return combined_summary


def get_routed_response(input_message="", knowledgeBase=None, filename=""):
    # Define RAG chain (QA over retrieved documents)
    llm = load_llm()
    similar_embeddings = knowledgeBase.similarity_search(input_message, k=10)
    general_context = format_docs(similar_embeddings)
    similar_embeddings = FAISS.from_documents(documents=similar_embeddings,
                                              embedding=OpenAIEmbeddings(
                                                  api_key=the_key))

    # creating the chain for integrating llm,prompt,stroutputparser
    retriever = similar_embeddings.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"  # 'stuff' means we use the LLM directly on retrieved content
    )

    # Define Summarization chain
    summarization_prompt = PromptTemplate(
        input_variables=["text"],
        template="{text}"
    )
    summarization_context = load_full_knowledge_base(filename=filename)
    summarization_chain = (
                        {"text": RunnablePassthrough()}
                        | summarization_prompt
                        | llm
                        | StrOutputParser()
                )

    task_type = router_func(input_text=(input_message))
    print(task_type.lower())
    # Route based on the LLM's decision
    if 'summarization' in task_type.lower():
        len(summarization_context.split())
        if len(summarization_context.split()) > MAX_TKN_LMT:
            print(f"=======given context has length {len(summarization_context.split())} which is exceeding the token limit. "
                  "recursive summarization is getting called.========")
            summarization_context = recursive_summarization(summarization_context, summarization_chain, max_tokens=5000,
                                                            min_length=4096)
            print(f"newly generated context with length {len(summarization_context.split())}: ", summarization_context)
        return summarization_chain.invoke(load_summarization_prompt(RFP_context=summarization_context,
                                                                    question=input_message))
    else:
        return rag_chain.invoke(load_summarization_prompt(RFP_context=general_context,
                                                                    question=input_message))['result']
