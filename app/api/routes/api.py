from fastapi import APIRouter

from app.api.routes import summarize

router = APIRouter()
router.include_router(summarize.router, tags=["summerizer"], prefix="/v1")
