import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader

from langchain.schema.document import Document#very important

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
def createVectorStore(pdf):
    if pdf is not None:
        print("execution started")
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        doc =  Document(page_content=text, metadata={"source": "local"})
        # print([doc])
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
      )
        chunks = text_splitter.split_documents([doc])
        # print(chunks)
        vector_store = Chroma.from_documents(chunks, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/temp_doc")
        print("Vector Store Created.......")
        #initially error was something like this that Chroma.from_documents and PdfReader returns text and hence 
        #expectation failed of doc with page_content whereas PyPDFLoader(ingest.py where readymade pdf file path is used)
        #returned document and hence acceptable but used file path and then tried out from_texts and worked in the sense(no error)
        #but wth docs they have made couldnt find both of the fucntion
    else:
        print("No PDF provided. Vector Store creation skipped.")