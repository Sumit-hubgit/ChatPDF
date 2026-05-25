from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document , UpdateMode
import hashlib
from langchain_community import(
    PyMuPDFLoader,
    DirectoryLoader
)
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)
load_dotenv()

class DocumentIngestion:

    def loadDoc():
        path = "data/pdf"
        directory_loader = DirectoryLoader(
            path,
            glob = "**/*.pdf",
            loader_cls = PyMuPDFLoader,
            show_progress = False
        )
        docs = directory_loader.load()
        return docs
    
    def chunking(docs:list[Document], chunkingModel:str)->list[Document]:
        chunking_model = HuggingFaceEmbeddings(
            model_name = chunkingModel
        )
        text_splitter = SemanticChunker(
            chunking_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        chunks = text_splitter.split_documents(docs)
        return chunks
    
    def vector_embedding(chunks:list[Document], embeddingModel):
        model = SentenceTransformer(embeddingModel)
        # for chunk in chunks:
        #     model.encode(chunk.page_content)
        for idx, chunk in enumerate(chunks):
            text = chunk.page_content
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)

                