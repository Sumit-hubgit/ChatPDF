from langchain_core.documents import Document
from vector_store import VectorStore
from config import Config
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self,chunks:list[Document], vectorStore: VectorStore, config:Config):
        self.chunks = chunks
        self.vectorStore = vectorStore
        self.config = config
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.reranker = CrossEncoder(config.reranker_model)

    def _rrf(self, bm25_docs:list[Document], vector_docs:list[Document]):
        scores:dict[str,float] = {}
        doc_map: dict[str, Document] = {}
        k = self.config.rrf_k

        for rank, doc in enumerate(bm25_docs):
            c = doc.page_content
            doc_map[c] = doc
            scores[c] = scores.get(c, 0) + 1 / (k + rank + 1)
        
        for rank, doc in enumerate(vector_docs):
            c = doc.page_content
            doc_map[c] = doc
            scores[c] = scores.get(c, 0) + 1 / (k + rank + 1)
            
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[c] for c, _ in ranked]