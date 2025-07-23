from fastapi import APIRouter
from .helpers import scan_directory_files
from ..core.common import DEFAULT_DATASETS_DIR
from ..core.api_response import ApiResponse

datasets_router = APIRouter(prefix="/datasets", tags=["datasets"])


@datasets_router.get("/")
async def get_datasets():
    """
    Get all dataset files by scanning the datasets directory.

    Returns:
        List[Dict]: Array of dataset files with filenames
    """
    try:
        files = scan_directory_files(DEFAULT_DATASETS_DIR)
        return ApiResponse.success(data=files)

    except Exception as e:
        raise ApiResponse.exception(
            status_code=500,
            api_message="Error getting datasets",
            exc=e,
            rethrow_if_exc_http_exception=True,
        ) from e