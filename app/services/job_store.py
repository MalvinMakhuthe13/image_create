from uuid import UUID

from app.schemas.jobs import JobResponse

# Development-only in-memory store. Replace with Supabase repository before deployment.
job_store: dict[UUID, JobResponse] = {}
