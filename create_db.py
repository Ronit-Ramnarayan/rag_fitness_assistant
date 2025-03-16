import shutil
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os

#Load enviromment variables
load_dotenv()

#Set OPENAI_API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']

#Load Documents/Data
DATA_PATH = "data"
CHROMA_PATH = "chroma"

def load_documents():
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents

#Split Text into Chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
    length_function=len,
    add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[0]

    print(document.page_content)
    print(document.metadata)

    return chunks


#Create ChromaDB to store chunks via vector embeddings
def createChromaDB(chunks):
    #Clear out the database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    #Create new DB from the documents
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

#Load the data, split it up into chunks, and save to ChromaDB
def generate_data_store():
    #Load Documents
    documents = load_documents()

    #Split Documents into Chunks
    documentChunks = split_text(documents)

    #Store chunks to ChromaDB via Vector Embeddings
    createChromaDB(documentChunks)

def main():
    generate_data_store() #Load data
    


if __name__ == "__main__":
    main()