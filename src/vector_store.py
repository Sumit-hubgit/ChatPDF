from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document , UpdateMode
from sentence_transformers import SentenceTransformer
import hashlib
from config import Config

class VectorStore:
    def __init__(self,config:Config):
        self.config = config
        self.client = QdrantClient(
            url = config.qdrant_url,
            api_key = config.qdrant_api_key
        )
        self.embdeddin_model = SentenceTransformer(self.config.embedding_model)
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
    
    def _exists(self, point_id: str) -> bool:
        return len(self.client.retrieve(
            collection_name=self.config.collection_name,
            ids=[point_id],
            with_vectors=False,
            with_payload=False,
        )) > 0
    
    def ingest(self, chunks:list[Document]):
        points = []
        for chunk in chunks:
            text = chunk.page_content
            source = chunk.metadata.get("souce","unknown")
            page = chunk.metadata("page",0)
            point_id = self._make_id(source,page,text)

            if self._exists(point_id):
                continue
            vector = self.embdeddin_model.encode(text).tolist()
            payload = {"text":text, "source":source, "page":page}
            points.append(
                PointStruct(
                    id = point_id,
                    vector = vector,
                    payload=payload
                )
            )
        if(len(points)>0):
            self.client.upsert(
                collection_name = self.config.collection_name,
                points = points
            )
            print(f"Inserted {len(points)} new chunks")
        return len(points)
    
    def search(self, query: str, top_k: int | None = None) -> list[str]:
        vector = self.embdeddin_model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=vector,
            limit=top_k or self.settings.top_k,
        )
        return [p.payload["text"] for p in results.points]