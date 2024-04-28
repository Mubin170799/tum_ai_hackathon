# Import Library
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import run


app = FastAPI()

class Query(BaseModel):
    query: str


@app.get('/')
def index():
    return {'message': 'Mercedes car Recommendation'}

@app.post("/chatbot")
async def get_answer(query: Query):
    response = run(query.query, verbose=False)
    return {"response": response}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)