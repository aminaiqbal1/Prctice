from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from Pinecone_class import pincoine_ob
from dotenv import load_dotenv
load_dotenv()
import os

# Streamlit styling
st.markdown(
    """
    <style>
    .text-strip { 
        background-color: #87CEEB; /* Sky blue color */
        padding: 5px; 
        border-radius: 10px; 
        color: black; /* Text color */
    }
    </style>
    <div class="text-strip">
        <p style="font-size:25px; font-weight:bold;"> 🧑 CHAT WITH BOOK 📖 </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar content
st.sidebar.markdown("<div class='sidebar-header'>🎓 Welcome, Dear Students! </div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-subheader'>How Can I Assist You? 🤖</div>", unsafe_allow_html=True)

# Load document
data_path = 'mlbook.pdf'  
loader = PyMuPDFLoader(data_path)
data = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
data_chunk = text_splitter.split_documents(data)

# Using embedding model
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is missing. Please set it in your .env file.")
    st.stop()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Correct embedding model
pincoine_ob.create_index(index_name="book-data", dimentions=1536)

# Check if the index is available
pincoine_ob.check_index(index="book-data")

# Insert data into Pinecone
pincoine_ob.insert_data_in_namespace(data_chunk, embeddings=embeddings, index_name="book-data", name_space="all-data")

# Using LLM
llm = OpenAI(api_key=api_key, model_name="gpt-4o-mini")

# Retrieval setup
vectordb = pincoine_ob.retrieve_from_namespace(index_name="book-data", embeddings=embeddings, name_space="all-data")
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
user_input = st.chat_input("Type your question here...")
# Retrieve documents based on user input
retrieved_docs = retriever.invoke(user_input)

# Debugging: print the type and content of retrieved_docs
st.write(f"retrieved_docs: {retrieved_docs}, type: {type(retrieved_docs)}")

# Ensure retrieved_docs is a list and properly formatted
if isinstance(retrieved_docs, list) and retrieved_docs:
    formatted_docs = format_docs(retrieved_docs)
else:
    formatted_docs = "No relevant context found."

# Debugging: print the type and content of formatted_docs
st.write(f"formatted_docs: {formatted_docs}, type: {type(formatted_docs)}")
