import uuid
import time
from typing import Dict, Any, Optional
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class JobRegistry:
    """In-memory asynchronous job status registry."""
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, task_type: str) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "task_type": task_type,
            "status": JobStatus.PENDING,
            "progress": 0,
            "result_image_id": None,
            "result_image_url": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        return job_id

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        result_image_id: Optional[str] = None,
        result_image_url: Optional[str] = None,
        error: Optional[str] = None,
    ):
        if job_id in self._jobs:
            if status is not None:
                self._jobs[job_id]["status"] = status
            if progress is not None:
                self._jobs[job_id]["progress"] = progress
            if result_image_id is not None:
                self._jobs[job_id]["result_image_id"] = result_image_id
            if result_image_url is not None:
                self._jobs[job_id]["result_image_url"] = result_image_url
            if error is not None:
                self._jobs[job_id]["error"] = error
            self._jobs[job_id]["updated_at"] = time.time()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

job_registry = JobRegistry()
