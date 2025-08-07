"""
Project files API endpoints.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import validate_project_directory


class FileInfo(BaseModel):
    """File information model."""
    name: str
    path: str
    size: int
    is_directory: bool
    extension: Optional[str] = None
    modified_time: float


class ProjectFilesResponse(BaseModel):
    """Project files response model."""
    project_dir: str
    total_files: int
    total_directories: int
    files: List[FileInfo]


def scan_project_directory() -> Dict[str, Any]:
    """
    Scan the project directory and return file information.
    
    Returns:
        Dictionary containing project files information
    """
    try:
        project_dir = validate_project_directory()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    files: List[FileInfo] = []
    total_files = 0
    total_directories = 0
    
    try:
        for item in project_dir.rglob("*"):
            try:
                # Skip hidden files and directories
                if any(part.startswith('.') for part in item.parts[len(project_dir.parts):]):
                    continue
                
                stat = item.stat()
                relative_path = str(item.relative_to(project_dir))
                
                file_info = FileInfo(
                    name=item.name,
                    path=relative_path,
                    size=stat.st_size,
                    is_directory=item.is_dir(),
                    extension=item.suffix.lower() if item.suffix else None,
                    modified_time=stat.st_mtime
                )
                
                files.append(file_info)
                
                if item.is_dir():
                    total_directories += 1
                else:
                    total_files += 1
                    
            except (OSError, PermissionError):
                # Skip files/directories we can't access
                continue
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning directory: {str(e)}")
    
    # Sort files: directories first, then by name
    files.sort(key=lambda x: (not x.is_directory, x.name.lower()))
    
    return {
        "project_dir": str(project_dir),
        "total_files": total_files,
        "total_directories": total_directories,
        "files": [file_info.model_dump() for file_info in files]
    }


def create_project_files_router() -> APIRouter:
    """Create and configure the project files router."""
    router = APIRouter(prefix="/api", tags=["project-files"])

    @router.get("/project-files", response_model=ProjectFilesResponse)
    async def get_project_files() -> Dict[str, Any]:
        """Get all files in the project directory."""
        return scan_project_directory()
    
    return router