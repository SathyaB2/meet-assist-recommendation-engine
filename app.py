# Input API
#    ↓
# Validation Layer
#    ↓
# User Type Detection
#    ├── New User → Cold-start ✅
#    └── Existing User → ML Propensity ✅
#             ↓
#       Product Scoring
#             ↓
#       GenAI Adjustment ✅
#             ↓
#       Ranking Engine
#             ↓
#       GenAI Explanation ✅
#             ↓
#         Final Output

# “We designed a hybrid recommendation engine that combines deterministic scoring, 
# machine learning–based personalization, and GenAI-powered reasoning to 
# deliver accurate and explainable recommendations for both new and existing users.”

from fastapi import FastAPI
from pydantic import BaseModel
from model3 import recommend_meet_assist
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

# ✅ ✅ UPDATED INPUT MODEL (ML SUPPORT ✅)
class PassengerRequest(BaseModel):
    departure: str
    arrival: str
    departure_datetime: str
    arrival_datetime: str
    loyalty_tier: str
    party_size: int

    # ✅ NEW FIELD FOR ML PROPENSITY
    passenger_id: str | None = None   # optional


# ✅ Health check
@app.get("/health")
def health():
    return {"status": "API running ✅"}


# ✅ Recommendation API
@app.post("/recommend")
def recommend(data: PassengerRequest):

    try:
        # ✅ Convert input to dict
        input_data = data.dict()

        # ✅ Debug log (optional but useful)
        user_type = "Existing (ML)" if input_data.get("passenger_id") else "New (Cold-start)"
        print(f"✅ Processing {user_type} user")

        # ✅ Call your main logic
        result = recommend_meet_assist(input_data)

        return {
            "status": "success",
            "user_type": user_type,
            "recommendations": result
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc()
        }

