from langchain_chroma import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
load_dotenv(find_dotenv())

urls = [
    "https://7knetwork.com/business-ideas-in-kashmir/",
    "https://en.wikipedia.org/wiki/Kashmir",
    "https://en.wikipedia.org/wiki/Gulmarg",
    "https://en.wikipedia.org/wiki/Pahalgam",
    "https://en.wikipedia.org/wiki/Dal_Lake",
]
udata= []
for url in urls:
    try:
        udata += WebBaseLoader(url).load()
        print(f"✓ Loaded: {url}")
    except Exception as e:
        print(f"✗ Failed: {url} → {e}")

txt_files = [
    "places.txt",       
    "itineraries.txt",  
    "agencies.txt",     
    "shops.txt",        
    "food.txt", 
    "data.txt",        
]

data = []
for fname in txt_files:
    path = os.path.join(BASE_DIR, "documents", fname)
    try:
        data += TextLoader(path).load()
        print(f"✓ Loaded: {fname}")
    except FileNotFoundError:
        print(f"✗ Missing: {fname} — skipping")

if not udata and not data:
    print("No data to embed. Exiting.")
    exit(1)

all_data = udata + data
print(f"\nTotal documents loaded: {len(all_data)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200)
chunks = splitter.split_documents(all_data)

# print(chunks)

try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
except Exception as e:
    print(f"Failed to load embedding model: {e}")
    exit(1)

try:
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    print(f"Stored {vectordb._collection.count()} chunks.")
except Exception as e:
    print(f"Failed to create vectorstore: {e}")
    exit(1)