import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dotenv import load_dotenv
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE, ".env"))

class Config:
    GROQ_API_KEY                = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL                  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL             = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM               = 384
    CHROMA_PERSIST_DIR          = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHUNK_SIZE                  = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP               = int(os.getenv("CHUNK_OVERLAP", 50))
    K_INIT                      = int(os.getenv("K_INIT", 20))
    K_RERANK                    = int(os.getenv("K_RERANK", 10))
    K_FINAL                     = int(os.getenv("K_FINAL", 5))
    SIMILARITY_THRESHOLD        = float(os.getenv("SIMILARITY_THRESHOLD", 0.10))
    REDIS_HOST                  = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT                  = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD              = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB                    = int(os.getenv("REDIS_DB", 0))
    REDIS_CACHE_TTL             = int(os.getenv("REDIS_CACHE_TTL", 86400))
    CACHE_SIMILARITY_THRESHOLD  = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.95))
    CACHE_MAX_L1                = 256
    TAVILY_API_KEY              = os.getenv("TAVILY_API_KEY", "")
    DEEP_RESEARCH_MAX_QUERIES   = int(os.getenv("DEEP_RESEARCH_MAX_QUERIES", 5))
    DEEP_RESEARCH_MAX_RESULTS   = int(os.getenv("DEEP_RESEARCH_MAX_RESULTS", 10))
    MAX_RETRIES                 = int(os.getenv("MAX_RETRIES", 3))
    MIN_QUALITY                 = float(os.getenv("MIN_QUALITY", 0.15))

cfg = Config()
