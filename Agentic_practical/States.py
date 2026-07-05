from typing import TypedDict

class State(TypedDict):
	topic: str
	summary: str
	score: int

#2) pydantic approach 
#it is good at data validation and type checking at runtime 
from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic : str 
    score : int 
    summary : str = ""

    @field_validator
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("score must be positive")
