import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docustra.api.routers import health, ingest, query
from docustra.core import configure_logging, get_settings

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="Docustra API",
    description="Enterprise Document Intelligence Platform — Advanced RAG Patterns",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Docustra API", "docs": "/docs"}


# Instrument LangChain with OpenTelemetry if tracing is enabled
if settings.enable_tracing:
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
    except Exception:
        pass  # Non-fatal: tracing is optional


def start():
    uvicorn.run(
        "docustra.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    start()
