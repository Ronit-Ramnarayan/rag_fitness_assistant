from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Welcome to the RAG-powered fitness assistant!"}

@app.post("/ask")
def ask_fitness_assistant(request: QueryRequest):
    # Placeholder response
    return {"response": f"AI-generated answer for: {request.query}"}
