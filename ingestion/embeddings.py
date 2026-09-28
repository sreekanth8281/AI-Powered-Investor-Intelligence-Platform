from functools import lru_cache
import logging
import os 

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and cache local embedding model.
    """

    model_name  = os.getenv(
        "EMBEDDING-MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    device = os.getenv("EMBEDDING_DEVICE","cpu")

    logger.info(
        "loading local embedding model: %s on %s",
        model_name,
        device
    )

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = {
                "device": device,
            
            },
            encode_kwargs = {
                "normalize_embeddings": True,
            }
        )

        logger.info("Embedding model loaded succesfully")

        return embeddings

    except Exception:
        logger.exception(
            "Failed to load embedding model: %s",
            model_name,
        )

        raise