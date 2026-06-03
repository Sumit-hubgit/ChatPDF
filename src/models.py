from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from langchain_huggingface import HuggingFaceEmbeddings

from .config import Config

config = Config()


class ModelManager:

    embedding_model = SentenceTransformer(
        config.embedding_model
    )

    reranker_model = CrossEncoder(
        config.reranker_model
    )

    chunking_model = HuggingFaceEmbeddings(
        model_name=config.chunking_model
    )