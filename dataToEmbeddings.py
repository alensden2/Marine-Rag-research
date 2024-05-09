# RAG 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import DirectoryLoader
import openai

openai.api_key = ''

# Loader 
loader = DirectoryLoader("data")
documents = loader.load()

# Text Splitter 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

print(texts)


# Embeddings Store 
import pickle
import faiss
from langchain_community.vectorstores import FAISS

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device" : "cpu"})

print(instructor_embeddings)


