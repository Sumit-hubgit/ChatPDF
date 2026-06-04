from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document , UpdateMode
from .config import Config
import hashlib
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DirectoryLoader
)
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)
from .models import ModelManager
import time
load_dotenv()

class DocumentIngestion:

    def __init__(self,config:Config):
        self.config = config

    def loadDoc(self):
        start = time.time()
        path = self.config.path
        directory_loader = DirectoryLoader(
            path,
            glob = "**/*.pdf",
            loader_cls = PyMuPDFLoader,
            show_progress = False
        )
        docs = directory_loader.load()
        print("Loaded docs in: ", time.time()-start)
        return docs
    
    def chunking(self, docs:list[Document])->list[Document]:
        start = time.time()
        chunking_model = ModelManager.chunking_model
        text_splitter = SemanticChunker(
            chunking_model,
            breakpoint_threshold_type= self.config.breakpoint_threshold_type,
            breakpoint_threshold_amount = self.config.breakpoint_threshold_amount
        )
        chunks = text_splitter.split_documents(docs)
        print("Chunking done in: ", time.time()-start)
        return chunks
    
    def load_and_chunk(self)->list[Document]:
        return self.chunking(self.loadDoc())