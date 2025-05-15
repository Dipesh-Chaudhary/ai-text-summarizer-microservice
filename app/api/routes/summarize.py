from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from app.services.summarizer import get_summary # Adjust import path as per your structure

router = APIRouter()

class TextToSummarize(BaseModel):
    text: str = Field(..., min_length=50, description="The text to be summarized.")
    max_length: int = Field(150, gt=20, description="Maximum length of the summary.")
    min_length: int = Field(30, gt=5, description="Minimum length of the summary.")

class SummaryResponse(BaseModel):
    original_text: str
    summary: str

@router.post("/summarize/", response_model=SummaryResponse, tags=["AI Summarizer"])
async def summarize_text_endpoint(payload: TextToSummarize = Body(...)):
    """
    Accepts text and returns a summarized version.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    summary = get_summary(payload.text, payload.max_length, payload.min_length)

    if "Error:" in summary: # Basic error check from our service
        raise HTTPException(status_code=500, detail=summary)

    return SummaryResponse(original_text=payload.text, summary=summary)