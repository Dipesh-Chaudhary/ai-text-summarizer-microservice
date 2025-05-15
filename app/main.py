from contextlib import asynccontextmanager
from transformers import pipeline
import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)
summarization_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading the ML model
    global summarization_pipeline
    try:
        logger.info("Loading summarization pipeline...")
        summarization_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6")
        logger.info("Summarization pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load summarization pipeline on startup: {e}")
        summarization_pipeline = None 
    yield
    # Cleaning up the ML models and releasing the resources
    logger.info("Closing summarization pipeline (if any)...")

app = FastAPI(lifespan=lifespan)


def get_summary(text: str, max_length: int = 150, min_length: int = 30) -> str:
    if not summarization_pipeline: 
        logger.error("Summarization pipeline is not available.")
        return "Error: Summarization model not available."
    try:
        summary_output = summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary_output[0]['summary_text']
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return f"Error: Could not generate summary. {e}"