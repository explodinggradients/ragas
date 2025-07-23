from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime
from ..core.app_configuration import AppConfig
from ..core.api_response import ApiResponse

app_config = AppConfig.create()


def scan_directory_files(directory_name: str) -> List[Dict[str, Union[str, float]]]:
    """
    Scan a directory and return list of files with metadata.
    
    Args:
        directory_name: Name of the directory to scan (e.g., 'datasets', 'experiments')
        
    Returns:
        List[Dict]: Array of files with filename and date information
        
    Raises:
        ApiResponse.exception: If project directory not configured or scanning fails
    """
    try:
        project_dir = app_config.get_project_dir()
        if not project_dir:
            raise ApiResponse.exception(
                status_code=404,
                api_message="Project directory not configured"
            )

        target_path = Path(project_dir) / directory_name
        files = []

        if target_path.exists() and target_path.is_dir():
            for item in target_path.iterdir():
                if item.is_file():
                    stat = item.stat()
                    files.append({
                        "filename": item.name,
                        "created_at": stat.st_ctime,
                        "modified_at": stat.st_mtime,
                        "size": stat.st_size
                    })

        # Sort by modified date (newest first)
        files.sort(key=lambda x: x["modified_at"], reverse=True)
        
        return files

    except Exception as e:
        raise ApiResponse.exception(
            status_code=500,
            api_message=f"Error scanning {directory_name} directory",
            exc=e,
            rethrow_if_exc_http_exception=True,
        ) from e