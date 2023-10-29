from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("Bangalore_trip.pdf")
print(loader.load())