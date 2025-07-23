from fastapi import APIRouter
from .datasets import datasets_router
from .experiments import experiments_router

api_router = APIRouter(prefix="/api")
api_router.include_router(datasets_router)
api_router.include_router(experiments_router)


@api_router.get("/livez")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "I'm alive!"}
