
from fastapi import FastAPI
from pydantic import BaseModel
from model import recommend_meet_assist
import traceback
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Meet & Assist Recommendation API")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ FINAL INPUT MODEL (LATEST ✅)
class PassengerRequest(BaseModel):
    departure: str
    arrival: str
    departure_datetime: str
    arrival_datetime: str
    loyalty_tier: str
    party_size: int


@app.get("/health")
def health():
    return {"status": "API running ✅"}


@app.post("/recommend")
def recommend(data: PassengerRequest):

    try:
        result = recommend_meet_assist(data.dict())
        return {"recommendations": result}

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
