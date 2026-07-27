from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from agent.graph import graph
from api.schemas import ResearchRequest, ResearchResponse
from api.streaming import stream_research_events

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI()


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    result = await graph.ainvoke({"query": req.query})
    return ResearchResponse(report=result["report"], sources=result["retrieved_docs"])


@app.post("/research/stream")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_research_events(req.query), media_type="text/event-stream"
    )
