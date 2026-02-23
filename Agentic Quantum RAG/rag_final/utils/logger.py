import logging, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
for n in ["sentence_transformers","transformers","chromadb","httpx","urllib3",
          "llama_index","openai","filelock","huggingface_hub","torch","nltk","tqdm","redis","aiohttp"]:
    logging.getLogger(n).setLevel(logging.ERROR)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    return logger
