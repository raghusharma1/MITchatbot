from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
import re
from my_list import my_list
from requests.exceptions import ChunkedEncodingError
import time


start = time.time()
documents = []
metadatalist=[]
page_contentlist = []
for link in my_list: 
    try:
        loader = WebBaseLoader(link) 
        docs = loader.load()
        for doc in docs:
            metadatalist.append(doc.metadata)
            page_contentlist.append(doc.page_content)
        documents.append(docs)
    except ChunkedEncodingError as e:
        print("ChunkedEncodingError occurred. Ignoring and continuing.")
page_contentlistcleaned = []
for texts in page_contentlist:
    docs_string = ''.join(texts)
    docs_string_cleaned = re.sub(r'\s{2,}', '\n', docs_string)
    soup = BeautifulSoup(docs_string_cleaned, 'html.parser')
    text = soup.get_text()
    page_contentlistcleaned.append(text)
# Initialize an empty list to store the Document objects
cleaned_docs = []

# Loop through the text_list and metadata_list to create Document objects
for text, metadata in zip(page_contentlistcleaned, metadatalist):
    document = Document(page_content=text, metadata=metadata)
    cleaned_docs.append(document)
print(cleaned_docs)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)  
documents = text_splitter.split_documents(cleaned_docs) 
# Define the path to the pre-trained model you want to use
model_path = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device': 'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

vector = FAISS.from_documents(documents, embeddings)
vector.save_local(folder_path="./faiss_db", index_name="Index")
end = time.time()
taken = end-start
print("Faiss index created")
print("time taken:", taken)
print(vector.index.ntotal)
