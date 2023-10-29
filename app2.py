import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from ingest2 import createVectorStore

# Define the function to create a vector store
    # Implementation of createVectorStore function from ingest.py
    # ...

# Check for PDF upload
st.title("Chat With Pdf Powered by Zephyr and HuggingFace")
st.write("Upload a PDF file to get answers to your questions.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Call createVectorStore function
    createVectorStore(uploaded_file)

    local_llm = "zephyr-7b-beta.Q5_K_S.gguf"
    
    config = {
        'max_new_tokens': 1024,
        'repetition_penalty': 1.1,
        'temperature': 0.1,
        'top_k': 50,
        'top_p': 0.9,
        'stream': True,
        'threads': int(os.cpu_count() / 2)
    }
    
    llm = CTransformers(
        model=local_llm,
        model_type="mistral",
        lib="avx2", #for CPU use
        **config
    )
    
    # st.write("LLM Initialized...")
    print("LLm Initialized")
    
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    load_vector_store = Chroma(persist_directory="stores/temp_doc", embedding_function=embeddings)
    retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})
    
    # st.write("######################################################################")
    
    # Add a text input for user prompts
    user_prompt = st.text_input("Enter your question or prompt:")
    
    # Process user prompt and retrieve answers
    if st.button("Get Answer"):
        if user_prompt:
            chain_type_kwargs = {"prompt": prompt}
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
            response = qa(user_prompt)
            st.write("Answer:")
            st.write(response)

