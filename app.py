from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_community.document_loaders import(
    TextLoader,
    PyMuPDFLoader,
    DirectoryLoader
)
import os

load_dotenv()

qdrant_url = os.getenv("CLUSTER_ENDPOINT")
api_key = os.getenv("QDRANT_API")
qdrant_client = QdrantClient(
    url= qdrant_url, 
    api_key=api_key,
)

print(qdrant_client.get_collections())