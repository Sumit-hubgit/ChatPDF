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

    #Recipocal Rak Fusion
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
    

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top = scored[: self.config.rerank_top_k]

        for i, (doc, score) in enumerate(top, 1):
            print(f"Rank {i} | Score: {score:.4f} | {doc.page_content[:80]}...")

        return [doc for doc, _ in top]
    
    def retrieve(self, query: str, query_vector: list[float]) -> list[Document]:
        bm25_docs = self.bm25.invoke(query)
        vector_docs = self.vectorStore.search(query_vector, self.settings.vector_top_k)
        fused = self._rrf(bm25_docs, vector_docs)
        return self._rerank(query, fused)

print("hello world!!")