from fastapi import FastAPI
from nlp_model import FakeNewsInference

app = FastAPI()

@app.post("/")
async def predict(headline: str):
    model = Inference()
    return model.predict(headline)