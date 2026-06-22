from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1)

class Source(BaseModel):
    id: str
    source: str
    text: str
    score: float

class ResearchResponse(BaseModel):
    report: str
    sources: list[Source]