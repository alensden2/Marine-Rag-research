from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

folder_path = "db"

embedding = FastEmbedEmbeddings()
chunks = ""

def textSplit(dirpath):
    global chunks  # Define chunks as a global variable
    loader = CSVLoader(dirpath)
    docs = loader.load_and_split()
    print(f"filename: {docs}")

    chunks = text_splitter.split_documents(docs)

def embeddingToVectorStore(chunks):
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=folder_path)
    print(f"chunks len={len(chunks)}")

textSplit('Team-alpha-fishes/data/final_dataset.csv')
embeddingToVectorStore(chunks)  # Pass chunks as an argument