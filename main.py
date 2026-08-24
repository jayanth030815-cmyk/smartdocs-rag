from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "SmartDocs is running"}


from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    return {"you_asked": request.question}