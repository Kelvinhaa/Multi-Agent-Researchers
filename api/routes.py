from fastapi import FastAPI
from agent.graph import graph
from api.schemas import ResearchRequest, ResearchResponse

app = FastAPI()


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    result = await graph.ainvoke({"query": req.query})
    return ResearchResponse(report=result["report"], sources=result["retrieved_docs"])
