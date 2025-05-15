from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Initializing the summarization pipeline

try:
    # Using a smaller, faster model for demonstration and resource constraint. For better quality, consider 'facebook/bart-large-cnn'.
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6")
    logger.info("Summarization pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load summarization pipeline: {e}")
    summarizer = None

def get_summary(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """
    Generates a summary for the given text.
    """
    if not summarizer:
        return "Error: Summarization model not available."
    if not text or not text.strip():
        return "Error: Input text cannot be empty."

    try:
        # you might need to chunk the text first for very long texts
        summary_output = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary_output[0]['summary_text']
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return "Error: Could not generate summary."