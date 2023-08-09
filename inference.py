from fastapi import FastAPI
from nlp_model import FakeNewsInference
from constants import Environment as E

app = FastAPI()

@app.post("/")
async def predict(headline: str):
    model = FakeNewsInference()
    return model.predict(headline)