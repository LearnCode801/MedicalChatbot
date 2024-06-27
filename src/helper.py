from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain_ai21 import AI21LLM
import os
from dotenv import load_dotenv
load_dotenv()
AI21LLM_API_KEY = os.environ.get('AI21LLM_API_KEY')

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents



#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

#Loading  different llm model
def load_local_llm():
    llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.7})
    
    return llm


def load_AI21Lab_llm():
    
    llm = AI21LLM(model="j2-ultra",api_key=AI21LLM_API_KEY,max_tokens=1024)
    return llm
