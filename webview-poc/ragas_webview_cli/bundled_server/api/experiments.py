from fastapi import APIRouter
from .helpers import scan_directory_files
from ..core.common import DEFAULT_EXPERIMENTS_DIR
from ..core.api_response import ApiResponse

experiments_router = APIRouter(prefix="/experiments", tags=["experiments"])


@experiments_router.get("/")
async def get_experiments():
    """
    Get all experiment files by scanning the experiments directory.

    Returns:
        List[Dict]: Array of experiment files with filenames
    """
    try:
        files = scan_directory_files(DEFAULT_EXPERIMENTS_DIR)
        return ApiResponse.success(data=files)

    except Exception as e:
        raise ApiResponse.exception(
            status_code=500,
            api_message="Error getting experiments",
            exc=e,
            rethrow_if_exc_http_exception=True,
        ) from e