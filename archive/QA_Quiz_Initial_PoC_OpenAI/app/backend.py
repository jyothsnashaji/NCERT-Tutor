from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
from scripts.rag_pipeline import get_rag_response, generate_quiz, parse_quiz_text

app = FastAPI()

class QuizRequest(BaseModel):
    grade: str
    subject: str
    topic: Optional[str] = None
    
class Query(BaseModel):
    grade: str
    subject: str
    question: str = ""
    history: Optional[List[Dict[str, str]]] = None

@app.post("/ask")
def ask(query: Query):
    try:
        answer = get_rag_response(query.question, query.grade, query.subject, query.history)
        return {"answer": answer}
    except Exception as e:
        print(f"[ERROR] Backend crashed: {e}")
        return {"answer": f"Error: {e}"}


@app.post("/quiz")
def quiz_endpoint(payload: QuizRequest):
    try:
        raw_quiz_text = generate_quiz(subject=payload.subject, grade=payload.grade, topic=payload.topic)
        structured_quiz = parse_quiz_text(raw_quiz_text)
        return {"quiz": structured_quiz}
    except Exception as e:
        print(f"[ERROR] Quiz generation failed: {e}")
        return {"quiz": f"Error: {e}"}