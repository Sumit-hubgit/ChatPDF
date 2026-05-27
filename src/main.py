from config import Config
from ingestion import DocumentIngestion
from vector_store import VectorStore
from cache import RedisCache
from retrieval import HybridRetriever, RAGPipeline

config = Config()
def build_pipeline(chunks) -> RAGPipeline:
    cache = RedisCache(Config)
    store = VectorStore(Config, cache)
    retriever = HybridRetriever(chunks, store, Config)
    return RAGPipeline(Config, retriever, cache, store)

def ingest():
    loader = DocumentIngestion(Config)
    chunks = loader.load_and_chunk()

    cache = RedisCache(Config)
    store = VectorStore(Config, cache)
    store.ingest(chunks)
    return chunks

if __name__ == "__main__":
    chunks = ingest()
    pipeline = build_pipeline(chunks)

    query = "What is Positional Encoding?"
    answer = pipeline.answer(query)
    print("\nFINAL ANSWER:\n", answer)