from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException

from app.schemas.jobs import CreateJobRequest, JobResponse
from app.services.job_store import job_store

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=201)
def create_job(payload: CreateJobRequest) -> JobResponse:
    job = JobResponse(
        id=uuid4(),
        status="queued",
        generation_mode=payload.generation_mode,
        message="Job accepted. Frame count will be planned after source analysis.",
    )
    job_store[job.id] = job
    return job


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: UUID) -> JobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
