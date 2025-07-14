"""
Dataset service for reading and serving CSV dataset files.
"""

import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException


def get_csv_data() -> Dict[str, Any]:
    """Read CSV data from samples_data directory."""
    # Get the project root directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    csv_file = project_root / "samples_data" / "evaluation_results.csv"
    
    if not csv_file.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"CSV file not found at: {csv_file}"
        )
    
    try:
        data = []
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = [dict(row) for row in reader]
        
        return {
            "filename": csv_file.name,
            "total_rows": len(data),
            "columns": list(data[0].keys()) if data else [],
            "data": data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error reading CSV file: {str(e)}"
        )


def create_datasets_router(data_dir: Optional[str] = None) -> APIRouter:
    """Create router for dataset operations."""
    router = APIRouter(tags=["datasets"])
    
    @router.get("/api/datasets")
    async def get_datasets():
        """Get evaluation dataset."""
        return get_csv_data()
    
    return router