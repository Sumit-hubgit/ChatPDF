from dotenv import load_dotenv
load_dotenv()

class Config:
    embedding_model = "all-MiniLM-L6-v2"
    chunking_model = "all-MiniLM-L6-v2"
    breakpoint_threshold_type = "percentile"
    breakpoint_threshold_amount=95
    path = "data/pdf"

