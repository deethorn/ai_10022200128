#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
LOGS_DIR = BASE_DIR / "logs"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
ASSETS_DIR = BASE_DIR / "assets"

STUDENT_NAME = "Chizota Diamond Chizzy"
INDEX_NUMBER = "10022200128"
REPOSITORY_NAME = "ai_10022200128"
PROJECT_TITLE = "Academic City RAG Chatbot"

CSV_FILE = DATA_DIR / "ghana_election_results.csv"
PDF_FILE = DATA_DIR / "2025_budget_statement.pdf"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
FINAL_CONTEXT_CHUNKS = 3

SUPPORTED_EXTENSIONS = [".csv", ".pdf"]
