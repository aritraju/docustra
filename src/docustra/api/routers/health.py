import httpx
import redis
from fastapi import APIRouter

from docustra.api.schemas import HealthResponse
from docustra.core import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    services: dict[str, str] = {}

    # Qdrant
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.qdrant_url}/healthz")
        services["qdrant"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        services["qdrant"] = "unreachable"

    # Redis
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "unreachable"

    # Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password)
        )
        driver.verify_connectivity()
        driver.close()
        services["neo4j"] = "ok"
    except Exception:
        services["neo4j"] = "unreachable"

    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)
