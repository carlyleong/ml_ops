# main.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Reddit Comment Classifier",
    version="0.1",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove."
)

class CommentInput(BaseModel):
    reddit_comment: str

@app.get("/")
def read_root():
    return {"message": "Hello from Reddit App!"}

@app.post("/predict")
def predict_comment(data: CommentInput):
    comment = data.reddit_comment.lower()

    # Dummy classifier logic — replace with model if needed
    if "hate" in comment or "offensive" in comment:
        return {"prediction": 1, "reason": "Toxic language detected"}
    return {"prediction": 0, "reason": "No action needed"}

