import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.output_parsers import StrOutputParser
import warnings
import os
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Load documents
loader = DirectoryLoader("./sample_pdfs_rag/clean_data/")
docs = loader.load()

# Prepare text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# Set up retriever
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Template
RAG_TEMPLATE = """
You are a multilingual assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}
"""

# Set up prompt template
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Load model
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)  # Increased token limit for better responses
hf = HuggingFacePipeline(pipeline=pipe)

# Set up QA chain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | hf
    | StrOutputParser()
)

# Streamlit app
st.set_page_config(page_title="Multilingual RAG Chatbot", page_icon="🤖")

# Title and Description
st.title("🤖 Multilingual RAG Chatbot")
st.markdown("Welcome to the Multilingual RAG Chatbot! Ask me anything about the content in the PDFs.")

# Sidebar with instructions
st.sidebar.title('🤖 Chatbot Instructions')
st.sidebar.markdown("""
- **Ask Questions**: Type in your question about the content of the PDFs.
- **Get Multilingual Responses**: I support multiple languages. Feel free to ask in your preferred language!
- **Clear Chat History**: Use the button to start a new conversation.
""")

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to display chat history
def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Clear chat history
def clear_chat_history():
    st.session_state.messages = []
    st.experimental_rerun()

# Clear chat button
st.button('Clear Chat', on_click=clear_chat_history)

# User input
if user_input := st.chat_input("Ask a question about the PDF content..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a new response
    with st.chat_message("assistant"):
        with st.spinner("Fetching response..."):
            response = qa_chain.invoke(user_input)
            response_text = response.get("result", "I don't know.")
            st.markdown(response_text)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Display chat history
display_chat()
