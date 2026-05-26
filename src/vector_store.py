from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document , UpdateMode
import hashlib
from config import Config

class VectorStore:
    def __init__(self,config:Config):
        self.config = config
        self.client = QdrantClient(
            url = config.qdrant_url,
            api_key = config.qdrant_api_key
        )
    def _ensure_collection(self)->None:
       collections = self.client.get_collections().collections
       collections_names = [c.name for c in collections]
       if self.config.collection_name not in collections_names:
           self.client.create_collection(
               collection_name = self.config.collection_name,
               vectors_config = VectorParams(
                   size = self.config.vector_size,
                   distance = self.config.vector_distance
               )
           )
    def _make_id(source:str, page:int, text:str)->str:
        return hashlib.md5(f"{source}_{page}_{text}".encode()).hexdigest()