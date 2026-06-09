from fastapi import FastAPI, UploadFile, File
import shutil
import os

from .config import Config
from .ingestion import DocumentIngestion
from .vector_store import VectorStore
from .retrieval import HybridRetriever, RAGPipeline
from .cache import RedisCache
from .awsconfig import AwsOperations

app = FastAPI()

config = Config()

UPLOAD_DIR = "data/pdf"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------------
# LOAD EVERYTHING ONLY ONCE
# -----------------------------------

print("Loading models and pipeline...")

cache = RedisCache(config)

store = VectorStore(config, cache)

loader = DocumentIngestion(config)

chunks = loader.load_and_chunk()

retriever = HybridRetriever(
    chunks,
    store,
    config
)

pipeline = RAGPipeline(
    config,
    retriever,
    cache,
    store
)

print("Pipeline loaded successfully")


# -----------------------------------
# HOME
# -----------------------------------

@app.get("/")
def home():
    return {"message": "ChatPDF API running 🚀"}


# -----------------------------------
# ASK QUESTION
# -----------------------------------

@app.post("/ask")
async def ask_question(query: str):

    answer = pipeline.answer(query)

    return {
        "query": query,
        "answer": answer
    }


# -----------------------------------
# UPLOAD PDF
# -----------------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Uploaded successfully"
    }

aws = AwsOperations(config)
@app.post("/upload-aws")
async def upload(file:UploadFile=File(...)):
    result = aws.upload_file(
        file.filename,
        file.file,
        folder="uploads"
    )
    return result

    