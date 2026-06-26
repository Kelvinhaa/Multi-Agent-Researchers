from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agent.graph import graph
from api.schemas import ResearchRequest, ResearchResponse
from api.streaming import stream_research_events

app = FastAPI()


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    result = await graph.ainvoke({"query": req.query})
    return ResearchResponse(report=result["report"], sources=result["retrieved_docs"])


@app.post("/research/stream")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    return StreamingResponse(stream_research_events(req.query), media_type="text/event-stream")
