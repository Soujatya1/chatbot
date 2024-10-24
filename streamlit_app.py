import streamlit as st
from llama_index import ServiceContext
from llama_index.vector_stores import VectorStoreIndex
from llama_index.readers import SimpleDirectoryReader
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings import HuggingFaceEmbeddings


uploaded_docs = st.file_uploader("Upload a file", type=["pdf"], accept_multiple_files=True)
documents = SimpleDirectoryReader('uploaded_docs').load_data()
llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

index=VectorStoreIndex.from_documents(documents,service_context=service_context)

query_engine=index.as_query_engine()

with input_box.container():
    prompt1 = st.text_input("Enter your question here.....", key="user_input", placeholder="Type your question...")

response=query_engine.query(prompt1)
st.write(response)
